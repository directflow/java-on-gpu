package ge.bog.gpu.utils;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static ge.bog.gpu.utils.Color.*;
import static ge.bog.gpu.utils.Color.CYAN;
import static ge.bog.gpu.utils.Color.ORANGE;
import static ge.bog.gpu.utils.Color.RESET;

public final class SystemInfo {

    private static final Logger LOGGER = LoggerFactory.getLogger(SystemInfo.class);

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private static final double HEAP_BUDGET_RATIO = 0.6;
    private static final int HOST_ARRAYS = 3;

    private static final int MAX_BACKENDS = 8;
    private static final int MAX_DEVICES = 16;

    private static final Map<String, int[]> CUDA_CORES = new LinkedHashMap<>();

    static {
        CUDA_CORES.put("L40S", new int[]{18176, 142});
        CUDA_CORES.put("L4", new int[]{7424, 58});
        CUDA_CORES.put("A10G", new int[]{9216, 80});
        CUDA_CORES.put("A100", new int[]{6912, 108});
        CUDA_CORES.put("V100", new int[]{5120, 80});
        CUDA_CORES.put("T4", new int[]{2560, 40});
    }

    private SystemInfo() {
    }

    public static List<String> environment(long size, String simd) {
        Runtime rt = Runtime.getRuntime();
        List<String> lines = new ArrayList<>();

        lines.add(String.format("CPU:     %s | %d threads | SIMD: %s", cpuModelAndSpeed(), rt.availableProcessors(), simd));
        lines.add(String.format("Heap:    max %,d MB", rt.maxMemory() / (1024 * 1024)));
        lines.addAll(gpuDevices());
        lines.add(String.format("OS:      %s %s (%s)",
                System.getProperty("os.name"), System.getProperty("os.version"), System.getProperty("os.arch")));
        lines.add(String.format("JVM:     %s %s", System.getProperty("java.vm.name"), System.getProperty("java.version")));
        lines.add(String.format("Params:  size=%,d", size));
        return lines;
    }

    public static String cpuModelAndSpeed() {
        String model = cpuModel();
        Double mhz = cpuMhz();
        // Intel/AMD names often already contain the speed ("... @ 2.50GHz")
        if (mhz != null && !model.toLowerCase(Locale.ROOT).contains("ghz")) {
            model += String.format(Locale.ROOT, " @ %.2f GHz", mhz / 1000.0);
        }
        return model;
    }

    public static String cpuModel() {
        String os = osName();
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
            model = lscpuField("Model name");
            if (isBlank(model)) {
                model = cpuinfoField("model name");
            }
            if (isBlank(model)) {
                model = cpuinfoField("hardware");
            }
        }

        return isBlank(model) ? System.getProperty("os.arch", "unknown") : model.replaceAll("\\s+", " ").trim();
    }

    private static Double cpuMhz() {
        String os = osName();
        if (os.contains("win")) {
            return null;
        }
        if (os.contains("mac")) {
            String hz = run("sysctl", "-n", "hw.cpufrequency_max"); // not available on Apple Silicon
            return parseDouble(hz) == null ? null : parseDouble(hz) / 1_000_000.0;
        }
        Double max = parseDouble(lscpuField("CPU max MHz"));
        return max != null ? max : parseDouble(cpuinfoField("cpu mhz"));
    }

    private static String lscpuField(String key) {
        String out = run("lscpu");
        if (out == null) {
            return null;
        }
        for (String line : out.split("\\R")) {
            if (line.startsWith(key + ":")) {
                return line.substring(key.length() + 1).trim();
            }
        }
        return null;
    }

    private static String cpuinfoField(String key) {
        try {
            for (String line : Files.readAllLines(Path.of("/proc/cpuinfo"))) {
                int idx = line.indexOf(':');
                if (idx > 0 && line.substring(0, idx).trim().toLowerCase(Locale.ROOT).equals(key)) {
                    return line.substring(idx + 1).trim();
                }
            }
        } catch (Exception ignored) {
        }
        return null;
    }

    public static List<String> gpuDevices() {
        List<String> lines = new ArrayList<>();

        String out = run("nvidia-smi", "--query-gpu=name,memory.total,clocks.max.sm", "--format=csv,noheader,nounits");
        if (out != null) {
            int i = 0;
            for (String line : out.split("\\R")) {
                String[] p = line.split(",");
                if (p.length < 3) {
                    continue;
                }
                String name = p[0].trim();
                Double mib = parseDouble(p[1].trim());
                Double mhz = parseDouble(p[2].trim());
                lines.add(String.format(Locale.ROOT, "GPU:     [%d] %s | %s | %s VRAM | max %s",
                        i++, name, cudaCores(name),
                        mib == null ? "?" : String.format(Locale.ROOT, "%.1f GiB", mib / 1024.0),
                        mhz == null ? "? GHz" : String.format(Locale.ROOT, "%.2f GHz", mhz / 1000.0)));
            }
        }

        // macOS: Apple GPU model and core count (memory is shared with the CPU)
        if (lines.isEmpty() && osName().contains("mac")) {
            String sp = run("system_profiler", "SPDisplaysDataType");
            if (sp != null) {
                String chip = null;
                String cores = null;
                for (String line : sp.split("\\R")) {
                    String t = line.trim();
                    if (t.startsWith("Chipset Model:")) {
                        chip = t.substring("Chipset Model:".length()).trim();
                    } else if (t.startsWith("Total Number of Cores:")) {
                        cores = t.substring("Total Number of Cores:".length()).trim();
                    }
                }
                if (chip != null) {
                    String mem = run("sysctl", "-n", "hw.memsize");
                    Double bytes = parseDouble(mem);
                    lines.add(String.format(Locale.ROOT, "GPU:     %s | %s GPU cores | %s unified memory",
                            chip, cores == null ? "?" : cores,
                            bytes == null ? "?" : String.format(Locale.ROOT, "%.0f GiB", bytes / (1024.0 * 1024 * 1024))));
                }
            }
        }

        if (lines.isEmpty()) {
            try {
                for (int b = 0; b < MAX_BACKENDS; b++) {
                    int found = 0;
                    for (int d = 0; d < MAX_DEVICES; d++) {
                        var device = TornadoExecutionPlan.getDevice(b, d);
                        if (device == null) {
                            break;
                        }
                        lines.add(String.format("GPU:     [%d:%d] %s", b, d, device.getDeviceName()));
                        found++;
                    }
                    if (found == 0) {
                        break;
                    }
                }
            } catch (Throwable ignored) {
            }
        }

        if (lines.isEmpty()) {
            lines.add("GPU:     not detected");
        }
        return lines;
    }

    private static String cudaCores(String gpuName) {
        for (Map.Entry<String, int[]> e : CUDA_CORES.entrySet()) {
            if (gpuName.contains(e.getKey())) {
                return String.format(Locale.ROOT, "%,d CUDA cores (%d SMs)", e.getValue()[0], e.getValue()[1]);
            }
        }
        return "cores n/a";
    }

    private static String osName() {
        return System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
    }

    private static String run(String... command) {
        try {
            ProcessBuilder pb = new ProcessBuilder(command).redirectErrorStream(true);
            pb.environment().put("LC_ALL", "C");
            Process p = pb.start();
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

    private static Double parseDouble(String s) {
        if (isBlank(s)) {
            return null;
        }
        try {
            return Double.parseDouble(s.trim());
        } catch (NumberFormatException e) {
            return null;
        }
    }

    private static boolean isBlank(String s) {
        return s == null || s.isBlank();
    }


    public static void printEnvironment(long size, int chunk, long chunks) {
        LOGGER.info("");
        LOGGER.info(YELLOW + "========== ENVIRONMENT ==========" + RESET);
        SystemInfo.environment(size, SPECIES.toString())
                .forEach(line -> LOGGER.info(YELLOW + line + RESET));
        LOGGER.info(YELLOW + String.format("Chunk:   %,d floats | executions: %,d", chunk, chunks) + RESET);
        LOGGER.info(YELLOW + "=================================" + RESET);
    }

    public static void report(long kernelNs, long endToEndNs, String title) {
        LOGGER.info("");
        LOGGER.info(ORANGE + "========== " + title + " ==========" + RESET);
        line("Kernel only", kernelNs);
        line("End-to-end", endToEndNs);

        LOGGER.info(ORANGE + "===============================" + RESET);
    }

    public static void line(String name, long ns) {
        LOGGER.info(String.format(CYAN + "%-21s total %,10.3f s" + RESET, name, ns / 1e9));
    }

    public static int fitChunkToHeap(int requested) {
        long budget = (long) (Runtime.getRuntime().maxMemory() * HEAP_BUDGET_RATIO);
        long maxChunk = budget / ((long) HOST_ARRAYS * Float.BYTES);
        int chunk = (int) Math.min(requested, Math.min(maxChunk, Integer.MAX_VALUE - 8));
        if (chunk < 1) {
            throw new IllegalStateException("Heap too small, increase -Xmx");
        }
        if (chunk < requested) {
            LOGGER.warn("Chunk reduced from {} to {} to fit heap", requested, chunk);
        }
        return chunk;
    }
}