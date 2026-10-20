package tornado;

import ge.bog.gpu.utils.MemoryTracker;
import ge.bog.gpu.utils.SystemInfo;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

class VectorApiVecMulTest {

    private static final long SIZE = 5_000_000_000L;

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private static final ByteOrder ORDER = ByteOrder.nativeOrder();

    private static void vecMul(MemorySegment a, MemorySegment b, MemorySegment c) {
        long i = 0;
        int width = SPECIES.length();
        long bound = SIZE - SIZE % width;

        for (; i < bound; i += width) {
            long byteOffset = i * Float.BYTES;

            FloatVector va = FloatVector.fromMemorySegment(SPECIES, a, byteOffset, ORDER);
            FloatVector vb = FloatVector.fromMemorySegment(SPECIES, b, byteOffset, ORDER);

            va.mul(vb).intoMemorySegment(c, byteOffset, ORDER);
        }

        for (; i < SIZE; i++) {
            long offset = i * Float.BYTES;

            float x = a.get(ValueLayout.JAVA_FLOAT, offset);
            float y = b.get(ValueLayout.JAVA_FLOAT, offset);

            c.set(ValueLayout.JAVA_FLOAT, offset, x * y);
        }
    }

    @Test
    void vecMulTest() {
        SystemInfo.printEnvironment(SIZE);

        long bytes = Math.multiplyExact(SIZE, Float.BYTES);

        MemoryTracker.track(tracker -> {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment a = arena.allocate(bytes, 64);
                MemorySegment b = arena.allocate(bytes, 64);
                MemorySegment c = arena.allocate(bytes, 64);

                a.fill((byte) 0);
                b.fill((byte) 0);
                c.fill((byte) 0);

                a.set(ValueLayout.JAVA_FLOAT, 0, 2.0f);
                b.set(ValueLayout.JAVA_FLOAT, 0, 3.0f);

                vecMul(a, b, c);

                long start = System.nanoTime();

                vecMul(a, b, c);

                tracker.mark();

                long elapsedNs = System.nanoTime() - start;

                SystemInfo.report(elapsedNs, "CPU Vector API");
            }
        });
    }
}