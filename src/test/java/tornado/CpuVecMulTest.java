package tornado;

import ge.bog.gpu.utils.MemoryTracker;
import ge.bog.gpu.utils.SystemInfo;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

class CpuVecMulTest {

    private static final int SIZE = 1_000_000_000;

    private static void vecMul(float[] a, float[] b, float[] c) {
        for (int i = 0; i < c.length; i++) {
            c[i] = a[i] * b[i];
        }
    }

    @Test
    void vecMulTest() {
        SystemInfo.printEnvironment(SIZE);

        MemoryTracker.track(tracker -> {
            float[] a = new float[SIZE];
            float[] b = new float[SIZE];
            float[] c = new float[SIZE];

            Arrays.fill(a, 2.0f);
            Arrays.fill(b, 3.0f);

            vecMul(a, b, c);

            long start = System.nanoTime();

            vecMul(a, b, c);

            tracker.mark();

            long elapsedNs = System.nanoTime() - start;
            SystemInfo.report(elapsedNs, "CPU Native Java");
        });
    }
}