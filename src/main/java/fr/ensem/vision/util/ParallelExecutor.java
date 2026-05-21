package fr.ensem.vision.util;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicInteger;

public final class ParallelExecutor {

    private ParallelExecutor() {}

    public static ExecutorService fixed(int threads, String poolName) {
        int n = Math.max(1, threads);
        ThreadFactory tf = new ThreadFactory() {
            private final AtomicInteger c = new AtomicInteger(0);
            @Override
            public Thread newThread(Runnable r) {
                Thread t = new Thread(r, poolName + "-" + c.incrementAndGet());
                t.setDaemon(true);
                return t;
            }
        };
        return Executors.newFixedThreadPool(n, tf);
    }

    public static int recommended(double cpuCap) {
        int cpus = Runtime.getRuntime().availableProcessors();
        return Math.max(1, (int) Math.floor(cpus * cpuCap));
    }
}
