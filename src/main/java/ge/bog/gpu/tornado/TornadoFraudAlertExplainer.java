package ge.bog.gpu.tornado;

import dev.langchain4j.model.jitllm.JitLLMChatModel;

import java.nio.file.Path;

/**
 * Explains a fraud ALERT in plain words with a local LLM:
 * LangChain4j → jitLLM (JitLLMChatModel) → TornadoVM → GPU.
 * <p>
 * Smallest practical model: Qwen3-0.6B-Q8_0.gguf (a GGUF file, about 0.6 GB).
 * The model runs on the GPU only when the JVM is started through TornadoVM
 * (-Duse.tornadovm=true); with onGPU(false) it runs on the CPU.
 */
public final class TornadoFraudAlertExplainer implements AutoCloseable {

    private final JitLLMChatModel model;

    public TornadoFraudAlertExplainer(Path modelPath, boolean onGpu) {
        this.model = JitLLMChatModel.builder()
                .modelPath(modelPath)
                .temperature(0.2)
                .maxTokens(200)
                .onGPU(onGpu)
                .think(false)
                .build();
    }

    public static boolean runOnGpu() {
        return Boolean.getBoolean("use.tornadovm");
    }

    public static String prompt(float amount, boolean abroad, boolean night, float score) {
        return """
                You are a bank fraud analyst. A payment was flagged by a scoring model.
                Facts: amount (scaled 0..1) = %.2f, paid abroad = %s, at night = %s, risk score = %.2f (alert above 2.0).
                Score formula: 0.8*amount + 1.5*abroad + 0.7*night.
                In two short sentences, explain why it was flagged and what the analyst should check.
                """.formatted(amount, abroad ? "yes" : "no", night ? "yes" : "no", score);
    }

    public String explain(float amount, boolean abroad, boolean night, float score) {
        return model.chat(prompt(amount, abroad, night, score));
    }

    @Override
    public void close() {
        try {
            model.close();
        } catch (Exception ignored) {
        }
    }
}
