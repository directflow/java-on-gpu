package ge.bog.gpu.babylon;

import ge.bog.gpu.utils.SystemInfo;
import hat.Accelerator;
import hat.Accelerator.Compute;
import hat.ComputeContext;
import hat.KernelContext;
import hat.NDRange;
import hat.backend.Backend;
import hat.buffer.F32Array;
import jdk.incubator.code.Reflect;

import java.lang.invoke.MethodHandles;
import java.util.concurrent.ThreadLocalRandom;

public class BabylonVecMul {

    private static final int LOCAL_SIZE = 256;

    @Reflect
    private static void vecMulKernel(F32Array a, F32Array b, F32Array c) {
        int i = KernelContext.GIX();
        if (i < c.length()) {
            c.array(i, a.array(i) * b.array(i));
        }
    }

    @Reflect
    private static void vecMul(ComputeContext cc, F32Array a, F32Array b, F32Array c, int global) {
        cc.dispatchKernel(NDRange.of1D(global, LOCAL_SIZE), () -> vecMulKernel(a, b, c));
    }

    public static void main(String[] args) {
        long size = Long.getLong("size", 1_000_000L);

        if (size <= 0) {
            throw new IllegalArgumentException("Vector size must be positive");
        }

        int chunk = (int) Math.min((long) Integer.getInteger("chunk", 1 << 26), size);
        long chunks = Math.ceilDiv(size, (long) chunk);

        SystemInfo.printEnvironment(size, chunk, chunks);

        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        F32Array a = F32Array.create(accelerator, chunk);
        F32Array b = F32Array.create(accelerator, chunk);
        F32Array c = F32Array.create(accelerator, chunk);

        for (int i = 0; i < chunk; i++) {
            a.array(i, i);
            b.array(i, 2.0f);
        }

        int global = (int) (Math.ceilDiv((long) chunk, LOCAL_SIZE) * LOCAL_SIZE);

        long executeNs = 0;
        long kernelNs = 0;

        for (long k = 0; k < chunks; k++) {
            long t = System.nanoTime();
            accelerator.compute((@Reflect Compute) cc -> vecMul(cc, a, b, c, global));
            executeNs += System.nanoTime() - t;

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

        SystemInfo.report(kernelNs, executeNs, "GPU (Babylon HAT)");
    }

    private static void check(F32Array a, F32Array b, F32Array c, int i) {
        float expected = a.array(i) * b.array(i);
        if (c.array(i) != expected) {
            throw new IllegalStateException("Mismatch at index " + i + ": expected " + expected + " but was " + c.array(i));
        }
    }
}
