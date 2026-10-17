package tornado;

import ge.bog.gpu.tornado.TornadoFraudAlertExplainer;
import ge.bog.gpu.tornado.TornadoFraudScoring;
import ge.bog.gpu.utils.PaymentBatchGenerator;
import ge.bog.gpu.utils.SystemInfo;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

import static ge.bog.gpu.utils.Color.CYAN;
import static ge.bog.gpu.utils.Color.ORANGE;
import static ge.bog.gpu.utils.Color.RESET;
import static ge.bog.gpu.utils.Color.YELLOW;
import static org.junit.jupiter.api.Assertions.*;

class TornadoFraudScoringTest {

    private static final Logger LOGGER = LoggerFactory.getLogger(TornadoFraudScoringTest.class);
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private static final float EPS = 1e-5f;

    private static final int DEFAULT_SIZE = 10;

    private static final int PRINT_LIMIT = 10;

    private static final double DEMO_FRAUD_RATE = 0.4;

    private static final int MIN_FRAUDS_FOR_QUALITY_GATE = 50;

    /**
     * -Dsize=10     -> number of payments (default 10); size <= 10 prints each payment and asks the LLM,
     *                  size > 10 prints only the summary
     * -Dfraud=true  -> random frauds are mixed in (demo rate for small batches, 1% for large ones)
     * -Dfraud=false -> no frauds (default)
     * -Dseed=123    -> reproduce a run, otherwise the seed is random
     * -Dmodel=/path/Qwen3-0.6B-Q8_0.gguf -> enables the LLM explanations
     */
    @Test
    void fraudScoringPositiveTest() throws Exception {
        int size = Integer.getInteger("size", DEFAULT_SIZE);
        assertTrue(size > 0, "size must be > 0");
        boolean verbose = size <= PRINT_LIMIT;

        printEnvironment(size);

        boolean withFraud = Boolean.getBoolean("fraud");
        long seed = Long.getLong("seed", System.nanoTime());

        PaymentBatchGenerator batch = verbose
                ? PaymentBatchGenerator.of(size, seed, withFraud ? DEMO_FRAUD_RATE : 0.0)
                : PaymentBatchGenerator.of(size, seed, withFraud);

        float[] weights = PaymentBatchGenerator.WEIGHTS;

        int actualFraud = 0;
        int predictedFraud = 0;
        int truePositive = 0;
        int falsePositive = 0;
        int falseNegative = 0;

        float[] scores;

        try (TornadoFraudScoring scorer = new TornadoFraudScoring(size, weights)) {
            scores = scorer.scoreBatch(batch);
            boolean[] labels = batch.isFraud();

            float[] expected = PaymentBatchGenerator.scoreReference(
                    batch.features(), weights, size, PaymentBatchGenerator.FEATURES);
            assertArrayEquals(expected, scores, EPS);

            assertArrayEquals(expected, scorer.scoreBatch(batch), EPS);

            for (int i = 0; i < size; i++) {
                boolean actual = labels[i];
                boolean predicted = PaymentBatchGenerator.isAlert(scores[i]);

                if (actual) {
                    actualFraud++;
                }
                if (predicted) {
                    predictedFraud++;
                }
                if (actual && predicted) {
                    truePositive++;
                } else if (!actual && predicted) {
                    falsePositive++;
                } else if (actual) {
                    falseNegative++;
                }
            }
        }

        LOGGER.info("");
        LOGGER.info(ORANGE + "Total transactions: {}", size);
        LOGGER.info(ORANGE + "Actual fraud:       {}", actualFraud);
        LOGGER.info(ORANGE + "Predicted fraud:    {}", predictedFraud);
        LOGGER.info(ORANGE + "True positives:     {}", truePositive);
        LOGGER.info(ORANGE + "False positives:    {}", falsePositive);
        LOGGER.info(ORANGE + "False negatives:    {}" + RESET, falseNegative);

        assertEquals(actualFraud, truePositive + falseNegative);
        assertEquals(predictedFraud, truePositive + falsePositive);
        if (!withFraud) {
            assertEquals(0, actualFraud, "no frauds expected when fraud flag is off");
        }

        if (actualFraud >= MIN_FRAUDS_FOR_QUALITY_GATE) {
            double recall = (double) truePositive / actualFraud;
            LOGGER.info(ORANGE + "Recall:             {}" + RESET, String.format(Locale.ROOT, "%.3f", recall));
            assertTrue(recall >= 0.6, "recall too low: " + recall);
        }

        if (verbose) {
            printCases(batch, scores);
            explainAll(batch, scores);
        } else {
            LOGGER.info(ORANGE + "Per-payment output skipped: size {} > {}" + RESET, size, PRINT_LIMIT);
        }
        LOGGER.info("");
    }

    private static void printEnvironment(long size) {
        LOGGER.info("");
        LOGGER.info(YELLOW + "========== ENVIRONMENT ==========" + RESET);
        SystemInfo.environment(size, SPECIES.toString())
                .forEach(line -> LOGGER.info(YELLOW + line + RESET));
        LOGGER.info(YELLOW + "=================================" + RESET);
    }

    private static final String TABLE_FORMAT = "%-3s | %-6s | %-6s | %-6s | %-5s | %-5s | %-9s | %-5s | %s";

    private static void printCases(PaymentBatchGenerator batch, float[] scores) {
        String header = String.format(Locale.ROOT, TABLE_FORMAT,
                "#", "Amount", "EUR", "Abroad", "Night", "Score", "Actual", "Model", "Result");

        StringBuilder sb = new StringBuilder("---------------- " + batch.size() + " generated payments ----------------");
        sb.append('\n').append(header);
        sb.append('\n').append("-".repeat(header.length()));

        for (int i = 0; i < batch.size(); i++) {
            Payment p = Payment.of(batch, scores, i);
            sb.append('\n').append(String.format(Locale.ROOT, TABLE_FORMAT,
                    i,
                    String.format(Locale.ROOT, "%.2f", p.amount),
                    String.format(Locale.ROOT, "%.0f", p.amount * PaymentBatchGenerator.MAX_AMOUNT_EUR),
                    yesNo(p.abroad),
                    yesNo(p.night),
                    String.format(Locale.ROOT, "%.2f", p.score),
                    p.fraud ? "FRAUD" : "NO FRAUD",
                    p.alert ? "ALERT" : "OK",
                    outcome(p.fraud, p.alert)));
        }
        LOGGER.info(ORANGE + "{}" + RESET, sb);
    }

    private static void explainAll(PaymentBatchGenerator batch, float[] scores) {
        String model = System.getProperty("model");
        if (model == null || !Files.exists(Path.of(model))) {
            LOGGER.warn(ORANGE + "LLM explanation skipped, model not found: {} (use -Dmodel=/path/model.gguf)" + RESET,
                    model);
            return;
        }

        boolean onGpu = TornadoFraudAlertExplainer.runOnGpu();

        String answer;
        try (TornadoFraudAlertExplainer explainer = new TornadoFraudAlertExplainer(Path.of(model), onGpu)) {
            answer = explainer.explainBatch(batch, scores);
        }

        assertFalse(answer == null || answer.isBlank(), "LLM returned an empty answer");

        LOGGER.info("");
        LOGGER.info(CYAN + "{}" + RESET, "---------------- LLM analysis of all " + batch.size()
                + " payments ----------------\n" + answer.strip());
    }

    private static String outcome(boolean fraud, boolean alert) {
        if (fraud) {
            return alert ? "TRUE POSITIVE" : "FALSE NEGATIVE";
        }
        return alert ? "FALSE POSITIVE" : "TRUE NEGATIVE";
    }

    private static String yesNo(boolean value) {
        return value ? "yes" : "no";
    }

    private record Payment(float amount, boolean abroad, boolean night, float score, boolean fraud, boolean alert) {

        static Payment of(PaymentBatchGenerator batch, float[] scores, int index) {
            int f = PaymentBatchGenerator.FEATURES;
            float[] x = batch.features();
            return new Payment(
                    x[index * f],
                    x[index * f + 1] == 1f,
                    x[index * f + 2] == 1f,
                    scores[index],
                    batch.isFraud()[index],
                    PaymentBatchGenerator.isAlert(scores[index]));
        }
    }
}