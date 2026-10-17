package ge.bog.gpu.tornado;

import ge.bog.gpu.utils.PaymentBatchGenerator;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

public final class TornadoFraudScoring implements AutoCloseable {

    private final int batchSize;
    private final FloatArray x;
    private final FloatArray w;
    private final FloatArray out;
    private final TornadoExecutionPlan plan;

    public TornadoFraudScoring(int batchSize, float[] weights) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be > 0");
        }
        Objects.requireNonNull(weights, "weights");
        if (weights.length != PaymentBatchGenerator.FEATURES) {
            throw new IllegalArgumentException("weights.length must be " + PaymentBatchGenerator.FEATURES);
        }

        this.batchSize = batchSize;
        x = new FloatArray(Math.multiplyExact(batchSize, PaymentBatchGenerator.FEATURES));
        w = FloatArray.fromArray(weights);
        out = new FloatArray(batchSize);

        TaskGraph tg = new TaskGraph("fraud")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, w)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x)
                .task("score", TornadoFraudScoring::scoreKernel, x, w, out, PaymentBatchGenerator.FEATURES)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        plan = new TornadoExecutionPlan(tg.snapshot());
    }

    public synchronized float[] scoreBatch(PaymentBatchGenerator batch) {
        if (batch.size() != batchSize) {
            throw new IllegalArgumentException(
                    "Expected batch of " + batchSize + " payments, got " + batch.size());
        }

        float[] src = batch.features();
        MemorySegment.copy(MemorySegment.ofArray(src), 0L, x.getSegment(), 0L,
                (long) src.length * Float.BYTES);

        plan.execute();

        return out.toHeapArray();
    }

    public static void scoreKernel(FloatArray x, FloatArray w, FloatArray out, int f) {
        for (@Parallel int t = 0; t < out.getSize(); t++) {
            float z = 0f;
            for (int k = 0; k < f; k++) {
                z += w.get(k) * x.get(t * f + k);
            }
            out.set(t, z);
        }
    }

    @Override
    public void close() throws TornadoExecutionPlanException {
        plan.close();
    }
}