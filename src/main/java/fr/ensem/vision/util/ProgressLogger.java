package fr.ensem.vision.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

public final class ProgressLogger implements AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(ProgressLogger.class);

    private final String stageLabel;
    private final EtaCalculator eta;
    private final ResourceMonitor resources;
    private final ScheduledExecutorService executor;
    private final long intervalSeconds;
    private ScheduledFuture<?> task;

    public ProgressLogger(String stageLabel, EtaCalculator eta, ResourceMonitor resources, long intervalSeconds) {
        this.stageLabel = stageLabel;
        this.eta = eta;
        this.resources = resources;
        this.intervalSeconds = Math.max(1L, intervalSeconds);
        this.executor = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "progress-logger");
            t.setDaemon(true);
            return t;
        });
    }

    public void start() {
        task = executor.scheduleAtFixedRate(this::logSnapshot, intervalSeconds, intervalSeconds, TimeUnit.SECONDS);
    }

    private void logSnapshot() {
        try {
            long processed = eta.getProcessed();
            long total = eta.getTotal();
            double fps = eta.fps();
            String etaStr = eta.formatEta();
            double cpu = resources == null ? Double.NaN : resources.cpuLoad();
            double mem = resources == null ? Double.NaN : resources.usedRamRatio();
            if (total > 0) {
                double pct = 100d * processed / (double) total;
                LOG.info("[{}] {}/{} ({}%) fps={} eta={} cpu={}% ram={}%",
                        stageLabel, processed, total,
                        String.format("%.1f", pct),
                        String.format("%.2f", fps), etaStr,
                        Double.isNaN(cpu) ? "?" : String.format("%.0f", cpu * 100d),
                        Double.isNaN(mem) ? "?" : String.format("%.0f", mem * 100d));
            } else {
                LOG.info("[{}] {} fps={} cpu={}% ram={}%",
                        stageLabel, processed,
                        String.format("%.2f", fps),
                        Double.isNaN(cpu) ? "?" : String.format("%.0f", cpu * 100d),
                        Double.isNaN(mem) ? "?" : String.format("%.0f", mem * 100d));
            }
            if (resources != null) resources.maybeFreeMemory();
        } catch (Exception ignored) {}
    }

    @Override
    public void close() {
        if (task != null) task.cancel(false);
        executor.shutdownNow();
        logSnapshot();
    }
}
