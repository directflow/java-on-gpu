package tornado;

import dev.langchain4j.model.jitllm.JitLLMChatModel;
import ge.bog.gpu.tornado.TornadoFraudAlertExplainer;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

import static ge.bog.gpu.utils.Color.ORANGE;
import static ge.bog.gpu.utils.Color.RESET;

class TornadoJitLLMTest {

    private static final Logger LOGGER = LoggerFactory.getLogger(TornadoJitLLMTest.class);

    private static final String QUESTION = "Explain GPU computing in three simple sentences.";

    @Test
    void questionAndAnswerTest() {
        String modelProperty = System.getProperty("model");

        if (Objects.isNull(modelProperty) || modelProperty.isBlank()) {
            throw new IllegalArgumentException("No model file path provided in system properties.");
        }

        Path modelPath = Path.of(modelProperty);

        if (!Files.exists(modelPath)) {
            throw new IllegalArgumentException("The specified model file does not exist: " + modelPath.toAbsolutePath());
        }

        LOGGER.info(ORANGE + "Question: " + QUESTION + RESET);

        try (JitLLMChatModel model = JitLLMChatModel.builder()
                .modelPath(modelPath)
                .temperature(0.7)
                .topP(0.9)
                .maxTokens(256)
                .onGPU(TornadoFraudAlertExplainer.runOnGpu())
                .build()) {

            String answer = model.chat(QUESTION);

            LOGGER.info(ORANGE + "Answer: {}" + RESET, answer);
        }
    }
}
