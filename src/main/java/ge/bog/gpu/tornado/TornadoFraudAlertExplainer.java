package ge.bog.gpu.tornado;

import dev.langchain4j.model.jitllm.JitLLMChatModel;
import ge.bog.gpu.utils.PaymentBatchGenerator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;

/**
 * Explains a scored payment in plain words with a local LLM:
 * LangChain4j → jitLLM (JitLLMChatModel) → TornadoVM → GPU.
 * Works for flagged payments (alerts) and for payments that were not flagged.
 * <p>
 * Smallest practical model: Qwen3-0.6B-Q8_0.gguf (a GGUF file, about 0.6 GB).
 * The model runs on the GPU only when the JVM is started through TornadoVM
 * (-Duse.tornadovm=true); with onGPU(false) it runs on the CPU.
 */
public final class TornadoFraudAlertExplainer implements AutoCloseable {

    private static final Logger LOGGER = LoggerFactory.getLogger(TornadoFraudAlertExplainer.class);

    private static final String PROMPT_TEMPLATE = """
            You are a bank fraud analyst. %s
            Facts: amount (scaled 0..1) = %.2f, paid abroad = %s, at night = %s, risk score = %.2f (alert at or above %.1f).
            Score contributions: amount %.2f + abroad %.2f + night %.2f.
            In two short sentences, %s
            """;

    private static final String BATCH_SUSPICIOUS_TEMPLATE = """
            You are a friendly bank assistant. You write to a normal person
            who has no technical background and does not know how the check works.
            An automatic check reviewed %d payments and found %d of them suspicious.
            The suspicious payments are:
            %s
            
            Write a short, friendly message in simple, everyday language.
            Tell only about the suspicious payments and do not mention the normal ones.
            Line 1: one sentence about how many payments look suspicious.
            Then exactly %d more lines, one for each suspicious payment, in the same order as the list above.
            Each of those lines is one plain sentence about what looks unusual and what to do.
            Do not write numbers, labels or bullet points at the start of any line.
            Do not use technical words, scores or formulas. Write nothing else.
            """;

    private static final String BATCH_CLEAN_TEMPLATE = """
            You are a friendly bank assistant. You write to a normal person
            who has no technical background and does not know how the check works.
            An automatic check reviewed %d payments and found nothing suspicious. There was no fraud.
            Write two short, friendly sentences in simple, everyday language saying that no fraud was found
            and that the payments look normal. Do not use technical words, scores or formulas. Write nothing else.
            """;

    private static final int MAX_TOKENS = 1024;

    private static final String FLAGGED_INTRO = "A payment was flagged by a scoring model.";
    private static final String CLEARED_INTRO = "A payment was scored by a model and was NOT flagged.";
    private static final String FLAGGED_TASK = "explain why it was flagged and what the analyst should check.";
    private static final String CLEARED_TASK = "explain why it was not flagged and whether anything still deserves a check.";

    private final JitLLMChatModel model;

    public TornadoFraudAlertExplainer(Path modelPath, boolean onGpu) {
        this.model = JitLLMChatModel.builder()
                .modelPath(modelPath)
                .temperature(0.2)
                .maxTokens(MAX_TOKENS)
                .onGPU(onGpu)
                .think(false)
                .build();
    }

    public static boolean runOnGpu() {
        return Boolean.getBoolean("use.tornadovm");
    }

    public static String prompt(float amount, boolean abroad, boolean night, float score) {
        float[] w = PaymentBatchGenerator.WEIGHTS;
        boolean flagged = PaymentBatchGenerator.isAlert(score);

        return String.format(Locale.ROOT, PROMPT_TEMPLATE,
                flagged ? FLAGGED_INTRO : CLEARED_INTRO,
                amount, yesNo(abroad), yesNo(night), score, PaymentBatchGenerator.THRESHOLD,
                w[0] * amount, abroad ? w[1] : 0f, night ? w[2] : 0f,
                flagged ? FLAGGED_TASK : CLEARED_TASK);
    }

    public static String batchPrompt(PaymentBatchGenerator batch, float[] scores) {
        int f = PaymentBatchGenerator.FEATURES;
        float[] x = batch.features();

        StringBuilder rows = new StringBuilder();
        int suspicious = 0;
        for (int i = 0; i < batch.size(); i++) {
            if (!PaymentBatchGenerator.isAlert(scores[i])) {
                continue;
            }
            suspicious++;
            float amount = x[i * f];
            boolean abroad = x[i * f + 1] == 1f;
            boolean night = x[i * f + 2] == 1f;
            rows.append(String.format(Locale.ROOT,
                    "- about %.0f EUR (%s amount), %s, %s%n",
                    amount * PaymentBatchGenerator.MAX_AMOUNT_EUR, amountSize(amount),
                    abroad ? "paid in another country" : "paid in the home country",
                    night ? "made at night" : "made during the day"));
        }

        if (suspicious == 0) {
            return String.format(Locale.ROOT, BATCH_CLEAN_TEMPLATE, batch.size());
        }
        return String.format(Locale.ROOT, BATCH_SUSPICIOUS_TEMPLATE,
                batch.size(), suspicious, rows.toString().stripTrailing(), suspicious);
    }

    private static String amountSize(float amount) {
        if (amount < 0.25f) {
            return "small";
        }
        return amount < 0.5f ? "medium" : "large";
    }

    private static String yesNo(boolean value) {
        return value ? "yes" : "no";
    }

    public String explain(float amount, boolean abroad, boolean night, float score) {
        return model.chat(prompt(amount, abroad, night, score));
    }

    public String explainBatch(PaymentBatchGenerator batch, float[] scores) {
        return formatBatchReport(model.chat(batchPrompt(batch, scores)), suspiciousIndices(batch, scores));
    }

    static List<Integer> suspiciousIndices(PaymentBatchGenerator batch, float[] scores) {
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < batch.size(); i++) {
            if (PaymentBatchGenerator.isAlert(scores[i])) {
                indices.add(i);
            }
        }
        return indices;
    }

    private static final Pattern LEADING_LABEL = Pattern.compile(
            "^(?:(?:#|payment\\s*#?)\\s*\\d+\\s*[:.)-]\\s*|[-*\u2022]\\s*)+", Pattern.CASE_INSENSITIVE);

    private static final Pattern SUMMARY_LABEL = Pattern.compile("^summary\\s*:\\s*", Pattern.CASE_INSENSITIVE);

    static String formatBatchReport(String raw, List<Integer> suspicious) {
        if (raw == null) {
            return "";
        }

        List<String> lines = new ArrayList<>();
        for (String line : raw.strip().split("\\R")) {
            String cleaned = LEADING_LABEL.matcher(line.strip()).replaceFirst("").strip();
            if (!cleaned.isEmpty()) {
                lines.add(cleaned);
            }
        }

        int n = suspicious.size();
        if (n == 0 || (lines.size() != n && lines.size() != n + 1)) {
            return String.join("\n", lines);
        }

        StringBuilder report = new StringBuilder();
        int first = 0;
        if (lines.size() == n + 1) {
            report.append("Summary: ").append(SUMMARY_LABEL.matcher(lines.get(0)).replaceFirst(""));
            first = 1;
        }
        for (int k = 0; k < n; k++) {
            if (!report.isEmpty()) {
                report.append('\n');
            }
            report.append('#').append(suspicious.get(k)).append(": ").append(lines.get(first + k));
        }
        return report.toString();
    }

    @Override
    public void close() {
        try {
            model.close();
        } catch (Exception e) {
            LOGGER.warn("Failed to close LLM model", e);
        }
    }
}