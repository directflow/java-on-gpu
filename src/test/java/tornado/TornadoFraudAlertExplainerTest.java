package tornado;

import ge.bog.gpu.tornado.TornadoFraudAlertExplainer;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;

import static ge.bog.gpu.utils.Color.ORANGE;
import static ge.bog.gpu.utils.Color.RESET;
import static org.junit.jupiter.api.Assertions.assertFalse;

class TornadoFraudAlertExplainerTest {

    private static final Logger LOGGER = LoggerFactory.getLogger(TornadoFraudAlertExplainerTest.class);

    @Test
    void explainsAlertTest() {
        Path modelPath = Path.of(System.getProperty("model"));
        try (TornadoFraudAlertExplainer explainer =
                     new TornadoFraudAlertExplainer(modelPath, TornadoFraudAlertExplainer.runOnGpu())) {
            String answer = explainer.explain(0.9f, true, true, 2.92f);
            LOGGER.info(ORANGE + "AI explanation: {}" + RESET, answer);
            assertFalse(answer.isBlank(), "the model should answer");
        }
    }
}
