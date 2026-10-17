package ge.bog.gpu.tornado;

import ge.bog.gpu.utils.PaymentBatchGenerator;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public final class TornadoFraudScoring implements AutoCloseable {

    private final FloatArray x;
    private final FloatArray w;
    private final FloatArray out;
    private final TornadoExecutionPlan plan;

    public TornadoFraudScoring(int batchSize, float[] weights) {
        x = new FloatArray(batchSize * PaymentBatchGenerator.FEATURES);
        w = FloatArray.fromArray(weights);
        out = new FloatArray(batchSize);

        TaskGraph tg = new TaskGraph("fraud")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, w)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x)
                .task("score", TornadoFraudScoring::scoreKernel, x, w, out, PaymentBatchGenerator.FEATURES)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        plan = new TornadoExecutionPlan(tg.snapshot());
    }

    public float[] scoreBatch(PaymentBatchGenerator batch) {
        float[] src = batch.features();
        for (int i = 0; i < src.length; i++) x.set(i, src[i]);
        plan.execute();
        float[] result = new float[batch.size()];
        for (int t = 0; t < batch.size(); t++) result[t] = out.get(t);
        return result;
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
