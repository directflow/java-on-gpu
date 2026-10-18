package babylon;

import ge.bog.gpu.utils.SystemInfo;
import hat.Accelerator;
import hat.NDRange;
import hat.backend.Backend;
import hat.buffer.F32Array;
import jdk.incubator.code.Reflect;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandles;

class BabylonVecMulTest {

    @Reflect
    private static void vecMulKernel(F32Array a, F32Array b, F32Array c) {
        int i = hat.KernelContext.GIX();

        if (i < c.length()) {
            c.array(i, a.array(i) * b.array(i));
        }
    }

    @Reflect
    private static void vecMul(hat.ComputeContext cc, F32Array a, F32Array b, F32Array c, int size) {
        cc.dispatchKernel(NDRange.of1D(size, 256), () -> vecMulKernel(a, b, c));
    }

    @Test
    void vecMulTest() {
        int size = Integer.getInteger("size", 1_000_000);

        if (size <= 0) {
            throw new IllegalArgumentException(
                    "Vector size must be positive");
        }

        int chunk = size;

        SystemInfo.printEnvironment(size, chunk, 1);

        var lookup = MethodHandles.lookup();
        var accelerator = new Accelerator(lookup, Backend.FIRST);

        F32Array a = F32Array.create(accelerator, size);
        F32Array b = F32Array.create(accelerator, size);
        F32Array c = F32Array.create(accelerator, size);

        for (int i = 0; i < size; i++) {
            a.array(i, i);
            b.array(i, 2.0f);
        }

        long executeNs = 0;
        long kernelNs = 0;

        long startNs = System.nanoTime();

        accelerator.compute(cc -> vecMul(cc, a, b, c, size));

        executeNs += System.nanoTime() - startNs;

        for (int i = 0; i < size; i++) {
            check(a, b, c, i);
        }

        SystemInfo.report(kernelNs, executeNs, "GPU (Babylon HAT)");
    }

    private static void check(F32Array a, F32Array b, F32Array c, int i) {
        float expected = a.array(i) * b.array(i);
        float actual = c.array(i);

        if (actual != expected) {
            throw new AssertionError(
                    "Mismatch at index " + i
                            + ": expected " + expected
                            + " but was " + actual);
        }
    }
}