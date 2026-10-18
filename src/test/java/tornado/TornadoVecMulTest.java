package tornado;

import ge.bog.gpu.utils.SystemInfo;
import org.junit.jupiter.api.Test;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.util.concurrent.ThreadLocalRandom;

class TornadoVecMulTest {

    private static void vecMul(FloatArray a, FloatArray b, FloatArray c) {
        for (@Parallel int i = 0; i < c.getSize(); i++) {
            c.set(i, a.get(i) * b.get(i));
        }
    }

    @Test
    void vecMulTest() throws TornadoExecutionPlanException {
        long size = Long.getLong("size", 1_000_000L);

        if (size <= 0) {
            throw new IllegalArgumentException("Vector size must be positive");
        }

        int chunk = (int) Math.min((long) Integer.getInteger("chunk", 1 << 26), size);
        long chunks = Math.ceilDiv(size, (long) chunk);

        SystemInfo.printEnvironment(size, chunk, chunks);

        FloatArray a = new FloatArray(chunk);
        FloatArray b = new FloatArray(chunk);
        FloatArray c = new FloatArray(chunk);

        for (int i = 0; i < chunk; i++) {
            a.set(i, i);
            b.set(i, 2.0f);
        }

        TaskGraph taskGraph = new TaskGraph("vecMulTaskGraph")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b)
                .task("vecMul", TornadoVecMulTest::vecMul, a, b, c)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, c);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();

        long executeNs = 0;
        long kernelNs = 0;
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(immutableTaskGraph)) {
            plan.withProfiler(ProfilerMode.SILENT);

            for (long k = 0; k < chunks; k++) {
                long t = System.nanoTime();
                TornadoExecutionResult result = plan.execute();
                executeNs += System.nanoTime() - t;
                kernelNs += result.getProfilerResult().getDeviceKernelTime();

                if (k == 0) {
                    for (int i = 0; i < chunk; i++) {
                        check(a, b, c, i);
                    }
                } else {
                    check(a, b, c, 0);
                    check(a, b, c, chunk - 1);
                    check(a, b, c, ThreadLocalRandom.current().nextInt(chunk));
                }
            }
        }

        SystemInfo.report(kernelNs, executeNs, "GPU (TornadoVM)");
    }

    private static void check(FloatArray a, FloatArray b, FloatArray c, int i) {
        float expected = a.get(i) * b.get(i);
        if (c.get(i) != expected) {
            throw new AssertionError("Mismatch at index " + i + ": expected " + expected + " but was " + c.get(i));
        }
    }
}