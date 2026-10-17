package ge.bog.gpu.utils;

import java.util.Random;

/**
 * Simulated Kafka batch of payments (same data as the "Batch Fraud Scoring from Kafka" slide).
 * Features per payment: [amount (0..1), abroad (0/1), night (0/1)].
 * About 1% of payments are fraud: large amount, often abroad, often at night.
 */
public record PaymentBatchGenerator(float[] features, boolean[] isFraud, int size) {

    public static final int FEATURES = 3;
    public static final float[] WEIGHTS = {0.8f, 1.5f, 0.7f};   // amount, abroad, night
    public static final float THRESHOLD = 2.0f;

    public static PaymentBatchGenerator of(int size, long seed, boolean includeFraud) {
        Random rnd = new Random(seed);
        float[] x = new float[size * FEATURES];
        boolean[] fraud = new boolean[size];

        for (int t = 0; t < size; t++) {
            fraud[t] = includeFraud && rnd.nextDouble() < 0.01;

            float amountEur = fraud[t]
                    ? 900 + rnd.nextFloat() * 1500
                    : rnd.nextFloat() * 400;

            boolean abroad = fraud[t]
                    ? rnd.nextDouble() < 0.85
                    : rnd.nextDouble() < 0.10;

            boolean night = fraud[t]
                    ? rnd.nextDouble() < 0.70
                    : rnd.nextDouble() < 0.15;

            x[t * FEATURES] = Math.min(amountEur / 2000f, 1f);
            x[t * FEATURES + 1] = abroad ? 1f : 0f;
            x[t * FEATURES + 2] = night ? 1f : 0f;
        }

        return new PaymentBatchGenerator(x, fraud, size);
    }

    public static PaymentBatchGenerator generate(int size, long seed) {
        Random rnd = new Random(seed);
        float[] x = new float[size * FEATURES];
        boolean[] fraud = new boolean[size];
        for (int t = 0; t < size; t++) {
            fraud[t] = rnd.nextDouble() < 0.01;
            float amountEur = fraud[t] ? 900 + rnd.nextFloat() * 1500 : rnd.nextFloat() * 400;
            boolean abroad = fraud[t] ? rnd.nextDouble() < 0.85 : rnd.nextDouble() < 0.10;
            boolean night = fraud[t] ? rnd.nextDouble() < 0.70 : rnd.nextDouble() < 0.15;
            x[t * FEATURES] = Math.min(amountEur / 2000f, 1f);
            x[t * FEATURES + 1] = abroad ? 1f : 0f;
            x[t * FEATURES + 2] = night ? 1f : 0f;
        }
        return new PaymentBatchGenerator(x, fraud, size);
    }

    public static float[] scoreReference(float[] x, float[] w, int n, int f) {
        float[] out = new float[n];
        for (int t = 0; t < n; t++) {
            float z = 0f;
            for (int k = 0; k < f; k++) z += w[k] * x[t * f + k];
            out[t] = z;
        }
        return out;
    }
}
