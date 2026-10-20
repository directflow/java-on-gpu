package ge.bog.gpu.utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.util.function.Consumer;

import static ge.bog.gpu.utils.Color.*;

public class MemoryTracker {

    private static final Logger LOGGER = LoggerFactory.getLogger(MemoryTracker.class);

    private static final MemoryMXBean MX = ManagementFactory.getMemoryMXBean();
    private static final double MB = 1024.0 * 1024.0;

    private final long baseHeap;
    private final long baseNonHeap;

    private MemoryTracker() {
        System.gc();
        this.baseHeap = heapUsed();
        this.baseNonHeap = nonHeapUsed();
        print("Before:", baseHeap, baseNonHeap);
    }

    public static void track(Consumer<MemoryTracker> body) {
        body.accept(new MemoryTracker());
        LOGGER.info("");
    }

    public void mark() {
        LOGGER.info("");
        print("After:", heapUsed(), nonHeapUsed());
    }

    private void print(String stage, long heap, long nonHeap) {
        String label = "Memory";
        String line = String.format("[%s] %-10s heap=%s | non-heap=%s", label, stage, fmt(heap, baseHeap), fmt(nonHeap, baseNonHeap));
        LOGGER.info("{}{}{}", GREEN, line, RESET);
    }

    private static String fmt(long now, long base) {
        if (now < 0 || base < 0) {
            return "n/a";
        }
        return String.format("%,.0f MB (%+,.0f)", now / MB, (now - base) / MB);
    }

    private static long heapUsed() {
        return MX.getHeapMemoryUsage().getUsed();
    }

    private static long nonHeapUsed() {
        return MX.getNonHeapMemoryUsage().getUsed();
    }
}