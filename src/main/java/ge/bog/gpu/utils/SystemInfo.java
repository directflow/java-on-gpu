package ge.bog.gpu.utils;

import uk.ac.manchester.tornado.api.TornadoExecutionPlan;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.TimeUnit;

public final class SystemInfo {

    private static final int MAX_BACKENDS = 8;
    private static final int MAX_DEVICES = 16;

    private SystemInfo() {
    }

    public static List<String> environment(long size, int chunk, int chunks, int warmup, String simd) {
        Runtime rt = Runtime.getRuntime();
        List<String> lines = new ArrayList<>();

        lines.add("CPU:     " + cpuModel());
        lines.add(String.format("Cores:   %d logical | SIMD: %s", rt.availableProcessors(), simd));
        lines.add(String.format("Heap:    max %,d MB", rt.maxMemory() / (1024 * 1024)));
        lines.addAll(gpuDevices());
        lines.add(String.format("OS:      %s %s (%s)",
                System.getProperty("os.name"), System.getProperty("os.version"), System.getProperty("os.arch")));
        lines.add(String.format("JVM:     %s %s", System.getProperty("java.vm.name"), System.getProperty("java.version")));
        lines.add(String.format("Params:  size=%,d | chunk=%,d | chunks=%,d | warmup=%d", size, chunk, chunks, warmup));
        return lines;
    }

    // ------------------------------------------------------------------ CPU

    public static String cpuModel() {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        String model = null;

        if (os.contains("win")) {
            String out = run("reg", "query",
                    "HKLM\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", "/v", "ProcessorNameString");
            if (out != null) {
                for (String line : out.split("\\R")) {
                    int idx = line.indexOf("REG_SZ");
                    if (line.contains("ProcessorNameString") && idx >= 0) {
                        model = line.substring(idx + "REG_SZ".length()).trim();
                        break;
                    }
                }
            }
            if (isBlank(model)) {
                model = System.getenv("PROCESSOR_IDENTIFIER");
            }
        } else if (os.contains("mac")) {
            model = run("sysctl", "-n", "machdep.cpu.brand_string");
        } else {
            model = linuxCpuModel();
        }

        return isBlank(model) ? System.getProperty("os.arch", "unknown") : model.replaceAll("\\s+", " ").trim();
    }

    private static String linuxCpuModel() {
        try {
            for (String line : Files.readAllLines(Path.of("/proc/cpuinfo"))) {
                String l = line.toLowerCase(Locale.ROOT);
                if (l.startsWith("model name") || l.startsWith("hardware") || l.startsWith("model\t")) {
                    int idx = line.indexOf(':');
                    if (idx >= 0 && idx + 1 < line.length()) {
                        return line.substring(idx + 1).trim();
                    }
                }
            }
        } catch (Exception ignored) {
            // fall through to lscpu
        }
        String out = run("lscpu");
        if (out != null) {
            for (String line : out.split("\\R")) {
                if (line.startsWith("Model name:")) {
                    return line.substring("Model name:".length()).trim();
                }
            }
        }
        return null;
    }

    public static List<String> gpuDevices() {
        List<String> lines = new ArrayList<>();
        try {
            for (int b = 0; b < MAX_BACKENDS; b++) {
                int found = 0;
                for (int d = 0; d < MAX_DEVICES; d++) {
                    var device = TornadoExecutionPlan.getDevice(b, d);
                    if (device == null) {
                        break;
                    }
                    String tag = (b == 0 && d == 0) ? " (default)" : "";
                    lines.add(String.format("GPU:     [%d:%d] %s%s", b, d, device.getDeviceName(), tag));
                    found++;
                }
                if (found == 0) {
                    break;
                }
            }
        } catch (Throwable ignored) {
            // no more devices, or TornadoVM runtime not initialised
        }

        if (lines.isEmpty()) {
            String out = run("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader");
            if (out != null) {
                for (String line : out.split("\\R")) {
                    if (!line.isBlank()) {
                        lines.add("GPU:     " + line.trim());
                    }
                }
            }
        }
        if (lines.isEmpty()) {
            lines.add("GPU:     not detected");
        }
        return lines;
    }

    private static String run(String... command) {
        try {
            Process p = new ProcessBuilder(command).redirectErrorStream(true).start();
            if (!p.waitFor(5, TimeUnit.SECONDS)) {
                p.destroyForcibly();
                return null;
            }
            if (p.exitValue() != 0) {
                return null;
            }
            String out = new String(p.getInputStream().readAllBytes()).trim();
            return out.isEmpty() ? null : out;
        } catch (Exception e) {
            return null;
        }
    }

    private static boolean isBlank(String s) {
        return s == null || s.isBlank();
    }
}