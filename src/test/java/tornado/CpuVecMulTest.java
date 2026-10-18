package tornado;

import ge.bog.gpu.utils.SystemInfo;
import org.junit.jupiter.api.Test;

class CpuVecMulTest {

    private static final int WARMUP_ITERATIONS = Integer.getInteger("warmup", 10);

    private static void vecMul(float[] a, float[] b, float[] c) {
        for (int i = 0; i < c.length; i++) {
            c[i] = a[i] * b[i];
        }
    }

    @Test
    void vecMulTest() {
        long size = Long.getLong("size", 1_000_000L);

        if (size <= 0) {
            throw new IllegalArgumentException("Vector size must be positive");
        }

        int requested = (int) Math.min((long) Integer.getInteger("chunk", 1 << 26), size);
        int chunk = SystemInfo.fitChunkToHeap(requested);
        long chunks = Math.ceilDiv(size, (long) chunk);

        SystemInfo.printEnvironment(size, chunk, chunks);

        float[] a = new float[chunk];
        float[] b = new float[chunk];
        float[] c = new float[chunk];

        for (int i = 0; i < chunk; i++) {
            a[i] = i;
            b[i] = 2.0f;
        }

        for (int i = 0; i < Math.max(1, WARMUP_ITERATIONS); i++) {
            vecMul(a, b, c);
        }

        long kernelNs = 0;
        long start = System.nanoTime();
        for (long k = 0; k < chunks; k++) {
            long t = System.nanoTime();
            vecMul(a, b, c);
            kernelNs += System.nanoTime() - t;
        }
        long endToEndNs = System.nanoTime() - start;

        for (int i = 0; i < chunk; i++) {
            float expected = a[i] * b[i];
            if (c[i] != expected) {
                throw new AssertionError("Mismatch at index " + i + ": expected " + expected + " but was " + c[i]);
            }
        }

        SystemInfo.report(kernelNs, endToEndNs, "CPU (Native Java)");
    }
}