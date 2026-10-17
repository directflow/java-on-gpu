package tornado;

import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;

import java.util.Objects;
import java.util.concurrent.ThreadLocalRandom;

import static ge.bog.gpu.utils.Color.*;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class TornadoVecMulTest {

    private static final Logger LOGGER = LoggerFactory.getLogger(TornadoVecMulTest.class);

    private static void vecMul(float[] a, float[] b, float[] c) {
        for (@Parallel int i = 0; i < c.length; i++) {
            c[i] = a[i] * b[i];
        }
    }

    @Test
    void vecMulTest() throws TornadoExecutionPlanException {
        int size = Objects.requireNonNullElse(Integer.getInteger("size"), 1000_000);

        if (size <= 0) {
            throw new IllegalArgumentException("Vector size must be positive");
        }

        float[] a = new float[size];
        float[] b = new float[size];
        float[] c = new float[size];
        float[] expected = new float[size];

        for (int i = 0; i < size; i++) {
            a[i] = i;
            b[i] = 2.0f;
            expected[i] = a[i] * b[i];
        }

        TaskGraph taskGraph = new TaskGraph("vecmul")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b)
                .task("multiply", TornadoVecMulTest::vecMul, a, b, c)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, c);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(immutableTaskGraph)) {
            plan.execute();
        }

        for (int i = 0; i < size; i++) {
            if (c[i] != a[i] * b[i]) {
                throw new AssertionError("Mismatch at index " + i);
            }
        }

        int idx = ThreadLocalRandom.current().nextInt(size);

        assertArrayEquals(expected, c, 0.0f, "GPU result differs from CPU result");

        assertEquals(a[idx] * b[idx], c[idx], 0.0f, "Mismatch at index " + idx);

        assertEquals(0.0f, c[0], 0.0f, "First element");
        assertEquals(2.0f * (size - 1), c[size - 1], 0.0f, "Last element");

        LOGGER.info("");
        LOGGER.info(CYAN + "========== Result ==========" + RESET);
        LOGGER.info(CYAN + "OK: c[{}] = {} (expected {})" + RESET, idx, c[idx], a[idx] * b[idx]);
        LOGGER.info(CYAN + "=================================" + RESET);
        LOGGER.info("");
    }
}
