package fr.ensem.vision.eval;

import fr.ensem.vision.inference.InferenceEngine;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

public final class FpsBenchmark {

    private static final Logger LOG = LoggerFactory.getLogger(FpsBenchmark.class);

    public static final class Result {
        public double fps;
        public double meanMs;
        public double p50Ms;
        public double p95Ms;
        public int samples;
    }

    public static Result runOnDirectory(InferenceEngine engine, Path imagesDir, int maxSamples) throws java.io.IOException {
        List<Path> paths = new ArrayList<>();
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(imagesDir, "*.{jpg,jpeg,png}")) {
            for (Path p : ds) paths.add(p);
        }
        Collections.sort(paths);
        if (maxSamples > 0 && paths.size() > maxSamples) paths = paths.subList(0, maxSamples);

        List<Double> latencies = new ArrayList<>(paths.size());
        long startNs = System.nanoTime();
        for (Path p : paths) {
            try (PointerScope scope = new PointerScope()) {
                Mat img = imread(p.toAbsolutePath().toString());
                if (img == null || img.empty()) continue;
                long s = System.nanoTime();
                engine.infer(img);
                long e = System.nanoTime();
                latencies.add((e - s) / 1e6d);
                img.release();
                img.close();
            }
        }
        long elapsedNs = System.nanoTime() - startNs;

        Collections.sort(latencies);
        Result r = new Result();
        r.samples = latencies.size();
        if (latencies.isEmpty()) return r;
        double sum = 0d;
        for (double v : latencies) sum += v;
        r.meanMs = sum / latencies.size();
        r.p50Ms = latencies.get(latencies.size() / 2);
        r.p95Ms = latencies.get(Math.min(latencies.size() - 1, (int) Math.round(latencies.size() * 0.95)));
        r.fps = r.samples / (elapsedNs / 1e9d);
        LOG.info("Bench {} samples={} fps={} mean={}ms p50={}ms p95={}ms",
                engine.name(), r.samples,
                String.format("%.2f", r.fps),
                String.format("%.1f", r.meanMs),
                String.format("%.1f", r.p50Ms),
                String.format("%.1f", r.p95Ms));
        return r;
    }
}
