package tornado;

import ge.bog.gpu.utils.MemoryTracker;
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

class TornadoVecMulTest {

    private static final int SIZE = 1_000_000_000;

    private static void vecMul(FloatArray a, FloatArray b, FloatArray c) {
        for (@Parallel int i = 0; i < c.getSize(); i++) {
            c.set(i, a.get(i) * b.get(i));
        }
    }

    @Test
    void vecMulTest() throws TornadoExecutionPlanException {
        SystemInfo.printEnvironment(SIZE);

        MemoryTracker.track(tracker -> {
            FloatArray a = new FloatArray(SIZE);
            FloatArray b = new FloatArray(SIZE);
            FloatArray c = new FloatArray(SIZE);

            a.init(2.0f);
            b.init(3.0f);
            c.init(0.0f);

            TaskGraph taskGraph = new TaskGraph("vecMulTaskGraph")
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b)
                    .task("vecMul", TornadoVecMulTest::vecMul, a, b, c)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, c);

            ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();

            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(immutableTaskGraph)) {
                plan.withProfiler(ProfilerMode.SILENT);

                plan.execute();

                long start = System.nanoTime();

                TornadoExecutionResult result = plan.execute();

                tracker.mark();

                long executeNs = System.nanoTime() - start;
                long kernelNs = result.getProfilerResult().getDeviceKernelTime();

                SystemInfo.report(kernelNs, executeNs, "GPU TornadoVM");
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
    }
}