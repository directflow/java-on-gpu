package tornado;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;

import java.util.Objects;
import java.util.Random;

import static ge.bog.gpu.utils.Color.*;

class TornadoVecMulTest {

    private static final Logger LOGGER = LoggerFactory.getLogger(TornadoVecMulTest.class);

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private static final int WARMUP_ITERATIONS = 5;

    @Test
    void vecMulBenchmarkTest() throws TornadoExecutionPlanException {
        int size = Objects.requireNonNullElse(Integer.getInteger("size"), 1000_000);

        if (size <= 0) {
            throw new IllegalArgumentException("Vector size must be positive");
        }

        float[] a = new float[size];
        float[] b = new float[size];
        float[] cJava = new float[size];
        float[] cVector = new float[size];
        float[] cTornado = new float[size];

        Random random = new Random(42);

        for (int i = 0; i < size; i++) {
            a[i] = random.nextFloat();
            b[i] = random.nextFloat();
        }

        // ---------------- CPU: Native Java ----------------

        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            multiplyJava(a, b, cJava);
        }

        long start = System.nanoTime();

        multiplyJava(a, b, cJava);

        long javaNs = System.nanoTime() - start;

        // ---------------- CPU: Vector API ----------------

        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            multiplyVectorAPI(a, b, cVector);
        }

        start = System.nanoTime();

        multiplyVectorAPI(a, b, cVector);

        long vectorNs = System.nanoTime() - start;

        // ---------------- GPU: TornadoVM ----------------

        TaskGraph taskGraph = new TaskGraph("tornado.TornadoVecMulTest")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b)
                .task("multiply", TornadoVecMulTest::multiplyTornado, a, b, cTornado)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, cTornado);

        ImmutableTaskGraph immutableTaskGraph =
                taskGraph.snapshot();

        long kernelNs;
        long totalNs;

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(immutableTaskGraph)) {

            plan.withProfiler(ProfilerMode.SILENT);

            for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                plan.execute();
            }

            start = System.nanoTime();

            TornadoExecutionResult result = plan.execute();

            totalNs = System.nanoTime() - start;

            kernelNs = result.getProfilerResult().getDeviceKernelTime();
        }

        LOGGER.info("");

        LOGGER.info(String.format(CYAN + "Vector size: %,d elements" + RESET, size));

        LOGGER.info(CYAN + "========== CPU ==========" + RESET);

        LOGGER.info(String.format(GREEN + "Native Java:          %10.3f ms" + RESET, javaNs / 1_000_000.0));

        LOGGER.info(String.format(BLUE + "Java Vector API:      %10.3f ms" + RESET, vectorNs / 1_000_000.0));

        LOGGER.info("");

        LOGGER.info(YELLOW + "========== GPU (TornadoVM) ==========" + RESET);

        LOGGER.info(String.format(RED + "GPU kernel execution:  %10.3f ms" + RESET, kernelNs / 1_000_000.0));

        LOGGER.info(String.format(GREEN + "Total plan execution:  %10.3f ms" + RESET, totalNs / 1_000_000.0));

        LOGGER.info(CYAN + "=====================================" + RESET);
    }

    private static void multiplyJava(float[] a, float[] b, float[] c) {
        for (int i = 0; i < a.length; i++) {
            c[i] = a[i] * b[i];
        }
    }

    private static void multiplyVectorAPI(float[] a, float[] b, float[] c) {
        int i = 0;
        int upperBound = SPECIES.loopBound(a.length);

        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i);

            va.mul(vb).intoArray(c, i);
        }

        for (; i < a.length; i++) {
            c[i] = a[i] * b[i];
        }
    }

    private static void multiplyTornado(float[] a, float[] b, float[] c) {
        for (@Parallel int i = 0; i < a.length; i++) {
            c[i] = a[i] * b[i];
        }
    }
}
