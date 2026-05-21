package fr.ensem.vision.util;

import java.time.Duration;
import java.util.ArrayDeque;
import java.util.Deque;

public final class EtaCalculator {

    private final int windowSize;
    private final Deque<Long> window;
    private long lastSampleNanos = -1L;
    private long totalItems = -1L;
    private long processed = 0L;

    public EtaCalculator(int windowSize) {
        if (windowSize <= 1) throw new IllegalArgumentException("windowSize must be > 1");
        this.windowSize = windowSize;
        this.window = new ArrayDeque<>(windowSize);
    }

    public void setTotal(long total) {
        this.totalItems = total;
    }

    public void setProcessed(long processed) {
        this.processed = processed;
    }

    public synchronized void tick() {
        long now = System.nanoTime();
        if (lastSampleNanos > 0) {
            long delta = now - lastSampleNanos;
            if (window.size() == windowSize) window.pollFirst();
            window.addLast(delta);
        }
        lastSampleNanos = now;
        processed++;
    }

    public synchronized double averageItemSeconds() {
        if (window.isEmpty()) return 0d;
        long sum = 0L;
        for (long v : window) sum += v;
        return (sum / (double) window.size()) / 1_000_000_000d;
    }

    public synchronized double fps() {
        double s = averageItemSeconds();
        return s <= 0d ? 0d : 1d / s;
    }

    public synchronized Duration estimatedRemaining() {
        if (totalItems <= 0 || processed >= totalItems || window.isEmpty()) return Duration.ZERO;
        double s = averageItemSeconds();
        long remaining = totalItems - processed;
        double seconds = remaining * s;
        return Duration.ofMillis((long) (seconds * 1000d));
    }

    public synchronized String formatEta() {
        Duration d = estimatedRemaining();
        long s = d.getSeconds();
        long h = s / 3600;
        long m = (s % 3600) / 60;
        long sec = s % 60;
        return String.format("%02d:%02d:%02d", h, m, sec);
    }

    public synchronized long getProcessed() { return processed; }
    public synchronized long getTotal() { return totalItems; }
}
