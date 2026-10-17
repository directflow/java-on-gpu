package tornado;

import ge.bog.gpu.utils.SystemInfo;
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

import static ge.bog.gpu.utils.Color.*;

class TornadoVecMulBenchmarkTest {

    private static final Logger LOGGER = LoggerFactory.getLogger(TornadoVecMulBenchmarkTest.class);
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private static final int REQUESTED_CHUNK = Integer.getInteger("chunk", 1 << 24);
    private static final int WARMUP_ITERATIONS = Integer.getInteger("warmup", 10);
    private static final double HEAP_BUDGET_RATIO = 0.6;
    private static final int HOST_ARRAYS = 5;
    private static final double TOLERANCE = 1e-6;

    private static final class Samples {
        final long[] ns;
        int n;

        Samples(int capacity) {
            ns = new long[capacity];
        }

        void add(long v) {
            ns[n++] = v;
        }

        long sum() {
            long s = 0;
            for (int i = 0; i < n; i++) s += ns[i];
            return s;
        }
    }

    @Test
    void vecMulBenchmarkTest() throws TornadoExecutionPlanException {
        long size = Objects.requireNonNullElse(Long.getLong("size"), 1000_000L);

        if (size <= 0) {
            throw new IllegalArgumentException("Vector size must be positive");
        }

        int chunk = fitChunkToHeap(REQUESTED_CHUNK);
        chunk -= chunk % SPECIES.length();
        int chunks = (int) Math.ceilDiv(size, (long) chunk);

        printEnvironment(size);

        float[] a = new float[chunk];
        float[] b = new float[chunk];
        float[] cJava = new float[chunk];
        float[] cVector = new float[chunk];
        float[] cTornado = new float[chunk];

        TaskGraph taskGraph = new TaskGraph("vecmul")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b)
                .task("multiply", TornadoVecMulBenchmarkTest::multiplyTornado, a, b, cTornado)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, cTornado);
        ImmutableTaskGraph immutable = taskGraph.snapshot();

        Samples java = new Samples(chunks);
        Samples vector = new Samples(chunks);
        Samples gpuTotal = new Samples(chunks);
        Samples gpuKernel = new Samples(chunks);

        double maxError = 0;

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(immutable)) {
            plan.withProfiler(ProfilerMode.SILENT);

            fill(a, b, 0);
            for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                multiplyJava(a, b, cJava);
                multiplyVectorAPI(a, b, cVector);
                plan.execute();
            }

            for (int k = 0; k < chunks; k++) {
                fill(a, b, (long) k * chunk);

                long t = System.nanoTime();
                multiplyJava(a, b, cJava);
                java.add(System.nanoTime() - t);

                t = System.nanoTime();
                multiplyVectorAPI(a, b, cVector);
                vector.add(System.nanoTime() - t);

                t = System.nanoTime();
                TornadoExecutionResult result = plan.execute();
                gpuTotal.add(System.nanoTime() - t);
                gpuKernel.add(result.getProfilerResult().getDeviceKernelTime());

                maxError = Math.max(maxError, maxAbsDiff(cJava, cVector));
                maxError = Math.max(maxError, maxAbsDiff(cJava, cTornado));
            }
        }

        report(java, vector, gpuTotal, gpuKernel);

        if (maxError > TOLERANCE) {
            throw new AssertionError("Results diverge, max abs error = " + maxError);
        }
    }

    private static void printEnvironment(long size) {
        LOGGER.info("");
        LOGGER.info(YELLOW + "========== ENVIRONMENT ==========" + RESET);
        SystemInfo.environment(size, SPECIES.toString())
                .forEach(line -> LOGGER.info(YELLOW + line + RESET));
        LOGGER.info(YELLOW + "=================================" + RESET);
    }

    private static int fitChunkToHeap(int requested) {
        long budget = (long) (Runtime.getRuntime().maxMemory() * HEAP_BUDGET_RATIO);
        long maxChunk = budget / ((long) HOST_ARRAYS * Float.BYTES);
        int chunk = (int) Math.min(requested, Math.min(maxChunk, Integer.MAX_VALUE - 8));
        if (chunk < SPECIES.length()) {
            throw new IllegalStateException("Heap too small, increase -Xmx");
        }
        if (chunk < requested) {
            LOGGER.warn("Chunk reduced from {} to {} to fit heap", requested, chunk);
        }
        return chunk;
    }

    private static void fill(float[] a, float[] b, long globalOffset) {
        for (int i = 0; i < a.length; i++) {
            long g = globalOffset + i;
            a[i] = ((g * 2654435761L) & 0xFFFFFF) / (float) (1 << 24);
            b[i] = (((g + 7) * 40503L) & 0xFFFFFF) / (float) (1 << 24);
        }
    }

    private static double maxAbsDiff(float[] x, float[] y) {
        double m = 0;
        for (int i = 0; i < x.length; i++) {
            m = Math.max(m, Math.abs(x[i] - y[i]));
        }
        return m;
    }

    private static void report(Samples java, Samples vector, Samples gpuTotal, Samples gpuKernel) {
        LOGGER.info("");

        LOGGER.info(CYAN + "========== CPU ==========" + RESET);
        line(CYAN, "Native Java", java);
        line(CYAN, "Java Vector API", vector);

        LOGGER.info("");

        LOGGER.info(GREEN + "========== GPU (TornadoVM) ==========" + RESET);
        line(GREEN, "GPU kernel only", gpuKernel);
        line(GREEN, "GPU end-to-end", gpuTotal);

        double overheadSec = (gpuTotal.sum() - gpuKernel.sum()) / 1e9;
        LOGGER.info(ORANGE + String.format("Transfer + launch overhead: %,.3f s (%.1f%% of end-to-end)",
                overheadSec, 100.0 * (gpuTotal.sum() - gpuKernel.sum()) / gpuTotal.sum()) + RESET);
        LOGGER.info(CYAN + "=====================================" + RESET);
    }

    private static void line(String color, String name, Samples s) {
        double totalSec = s.sum() / 1e9;
        LOGGER.info(String.format(color + "%-16s total %,10.3f s" + RESET, name, totalSec));
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
            FloatVector.fromArray(SPECIES, a, i)
                    .mul(FloatVector.fromArray(SPECIES, b, i))
                    .intoArray(c, i);
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