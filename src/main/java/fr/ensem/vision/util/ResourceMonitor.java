package fr.ensem.vision.util;

import com.sun.management.OperatingSystemMXBean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.management.ManagementFactory;

public final class ResourceMonitor {

    private static final Logger LOG = LoggerFactory.getLogger(ResourceMonitor.class);

    private final OperatingSystemMXBean os;
    private final double cpuCap;
    private final double ramCap;
    private long lastGcMillis = 0;

    public ResourceMonitor(double cpuCap, double ramCap) {
        this.cpuCap = cpuCap;
        this.ramCap = ramCap;
        this.os = (OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();
    }

    public double cpuLoad() {
        double v = os.getCpuLoad();
        if (v < 0) v = os.getSystemLoadAverage();
        return v;
    }

    public double usedRamRatio() {
        Runtime r = Runtime.getRuntime();
        long total = r.totalMemory();
        long free = r.freeMemory();
        long used = total - free;
        long max = r.maxMemory();
        return max <= 0 ? 0d : (used / (double) max);
    }

    public int recommendedThreads() {
        int cpus = Runtime.getRuntime().availableProcessors();
        return Math.max(1, (int) Math.floor(cpus * cpuCap));
    }

    public void maybeFreeMemory() {
        double used = usedRamRatio();
        if (used > ramCap) {
            long now = System.currentTimeMillis();
            if (now - lastGcMillis > 5_000L) {
                LOG.warn("RAM usage {}% over cap {}%, triggering GC",
                        String.format("%.0f", used * 100d), String.format("%.0f", ramCap * 100d));
                System.gc();
                lastGcMillis = now;
            }
        }
    }

    public double cpuCap() { return cpuCap; }
    public double ramCap() { return ramCap; }
}
