package tornado;

import ge.bog.gpu.tornado.TornadoFraudScoring;
import ge.bog.gpu.utils.PaymentBatchGenerator;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static ge.bog.gpu.utils.Color.ORANGE;
import static ge.bog.gpu.utils.Color.RESET;

class TornadoFraudScoringTest {

    private static final Logger LOGGER = LoggerFactory.getLogger(TornadoFraudScoringTest.class);

    @Test
    void fraudScoringPositiveTest() throws Exception {
        int size = 10_000;

        PaymentBatchGenerator batch = PaymentBatchGenerator.of(size, 42L, Boolean.getBoolean("fraud"));

        float[] weights = {0.8f, 1.5f, 0.7f};
        float threshold = 2.0f;

        int actualFraud = 0;
        int predictedFraud = 0;
        int truePositive = 0;
        int falsePositive = 0;
        int falseNegative = 0;

        try (TornadoFraudScoring scorer = new TornadoFraudScoring(size, weights)) {
            float[] scores = scorer.scoreBatch(batch);
            boolean[] labels = batch.isFraud();

            for (int i = 0; i < size; i++) {
                boolean actual = labels[i];
                boolean predicted = scores[i] >= threshold;

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

        LOGGER.info(ORANGE + "Total transactions: {}", size);
        LOGGER.info(ORANGE + "Actual fraud:       {}", actualFraud);
        LOGGER.info(ORANGE + "Predicted fraud:    {}", predictedFraud);
        LOGGER.info(ORANGE + "True positives:     {}", truePositive);
        LOGGER.info(ORANGE + "False positives:    {}", falsePositive);
        LOGGER.info(ORANGE + "False negatives:    {}" + RESET, falseNegative);
    }
}
