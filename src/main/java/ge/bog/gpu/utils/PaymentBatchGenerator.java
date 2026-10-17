package ge.bog.gpu.utils;

import java.util.Objects;
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

    private static final double FRAUD_RATE = 0.01;
    public static final float MAX_AMOUNT_EUR = 2000f;

    private static final float LEGIT_MAX_EUR = 400f;
    private static final float FRAUD_MIN_EUR = 900f;
    private static final float FRAUD_RANGE_EUR = 1500f;

    private static final double LEGIT_ABROAD = 0.10;
    private static final double FRAUD_ABROAD = 0.85;
    private static final double LEGIT_NIGHT = 0.15;
    private static final double FRAUD_NIGHT = 0.70;

    public PaymentBatchGenerator {
        Objects.requireNonNull(features, "features");
        Objects.requireNonNull(isFraud, "isFraud");
        if (isFraud.length != size || features.length != size * FEATURES) {
            throw new IllegalArgumentException("Inconsistent batch: size=" + size
                    + ", features=" + features.length + ", labels=" + isFraud.length);
        }
    }

    public static boolean isAlert(float score) {
        return score >= THRESHOLD;
    }

    public static PaymentBatchGenerator of(int size, long seed, boolean includeFraud) {
        return of(size, seed, includeFraud ? FRAUD_RATE : 0.0);
    }

    public static PaymentBatchGenerator of(int size, long seed, double fraudRate) {
        if (fraudRate < 0.0 || fraudRate > 1.0) {
            throw new IllegalArgumentException("fraudRate must be in 0..1");
        }
        Random rnd = new Random(seed);
        float[] x = new float[size * FEATURES];
        boolean[] fraud = new boolean[size];

        for (int t = 0; t < size; t++) {
            double fraudDraw = rnd.nextDouble();
            fraud[t] = fraudDraw < fraudRate;

            float amountEur = fraud[t]
                    ? FRAUD_MIN_EUR + rnd.nextFloat() * FRAUD_RANGE_EUR
                    : rnd.nextFloat() * LEGIT_MAX_EUR;

            boolean abroad = rnd.nextDouble() < (fraud[t] ? FRAUD_ABROAD : LEGIT_ABROAD);
            boolean night = rnd.nextDouble() < (fraud[t] ? FRAUD_NIGHT : LEGIT_NIGHT);

            int base = t * FEATURES;
            x[base] = Math.min(amountEur / MAX_AMOUNT_EUR, 1f);
            x[base + 1] = abroad ? 1f : 0f;
            x[base + 2] = night ? 1f : 0f;
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