package fr.ensem.vision.eval;

import fr.ensem.vision.config.ClassMapping;
import fr.ensem.vision.detection.BoundingBox;
import fr.ensem.vision.detection.Detection;
import fr.ensem.vision.inference.InferenceEngine;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

public final class Evaluator {

    private static final Logger LOG = LoggerFactory.getLogger(Evaluator.class);

    public static final class Report {
        public String engine;
        public Map<String, Double> apPerClass;
        public double mapAt50;
        public double mapAt5095;
        public Map<String, PrecisionRecallF1> prf1PerClass;
        public int totalSamples;
        public double seconds;
    }

    private final InferenceEngine engine;
    private final ClassMapping mapping;

    public Evaluator(InferenceEngine engine, ClassMapping mapping) {
        this.engine = engine;
        this.mapping = mapping;
    }

    public Report run(Path imagesDir, Path labelsDir, int maxSamples) throws IOException {
        List<MapCalculator.Sample> samples = new ArrayList<>();
        Map<Integer, int[]> classTpFpFn = new HashMap<>();

        List<Path> imgs = new ArrayList<>();
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(imagesDir, "*.{jpg,jpeg,png}")) {
            for (Path p : ds) imgs.add(p);
        }
        imgs.sort((a, b) -> a.getFileName().toString().compareTo(b.getFileName().toString()));
        if (maxSamples > 0 && imgs.size() > maxSamples) imgs = imgs.subList(0, maxSamples);

        long startNs = System.nanoTime();
        for (Path img : imgs) {
            try (PointerScope scope = new PointerScope()) {
                Mat m = imread(img.toAbsolutePath().toString());
                if (m == null || m.empty()) continue;
                int w = m.cols();
                int h = m.rows();
                Path labelFile = labelsDir.resolve(stripExt(img.getFileName().toString()) + ".txt");
                List<Detection> gt = readGroundTruth(labelFile, w, h);
                MapCalculator.ImageGt gtSample = new MapCalculator.ImageGt(gt, w, h);

                List<Detection> preds = engine.infer(m);
                samples.add(new MapCalculator.Sample(gtSample, preds));

                accumulateMatch(gt, preds, classTpFpFn, 0.5d);

                m.release();
                m.close();
            }
        }
        long elapsedNs = System.nanoTime() - startNs;

        MapCalculator mapC = new MapCalculator(mapping.size());
        Map<Integer, Double> apMap = mapC.averagePrecisionPerClass(samples, 0.5d);
        double map50 = mapC.meanAveragePrecision(samples, 0.5d);
        double map5095 = mapC.mapRange(samples, 0.5d, 0.95d, 0.05d);

        Report r = new Report();
        r.engine = engine.name();
        r.apPerClass = new LinkedHashMap<>();
        for (Map.Entry<Integer, Double> e : apMap.entrySet()) {
            r.apPerClass.put(mapping.labelForId(e.getKey()), e.getValue());
        }
        r.mapAt50 = map50;
        r.mapAt5095 = map5095;
        r.prf1PerClass = new LinkedHashMap<>();
        for (Map.Entry<Integer, int[]> e : classTpFpFn.entrySet()) {
            int[] tpfp = e.getValue();
            r.prf1PerClass.put(mapping.labelForId(e.getKey()),
                    new PrecisionRecallF1(tpfp[0], tpfp[1], tpfp[2]));
        }
        r.totalSamples = samples.size();
        r.seconds = elapsedNs / 1e9d;
        LOG.info("Eval {} samples={} mAP50={} mAP50-95={}", engine.name(), r.totalSamples,
                String.format("%.3f", r.mapAt50), String.format("%.3f", r.mapAt5095));
        return r;
    }

    private void accumulateMatch(List<Detection> gt, List<Detection> preds,
                                 Map<Integer, int[]> classMap, double iouThreshold) {
        for (Detection g : gt) classMap.computeIfAbsent(g.getClassId(), k -> new int[3]);
        for (Detection p : preds) classMap.computeIfAbsent(p.getClassId(), k -> new int[3]);

        boolean[] usedGt = new boolean[gt.size()];
        for (Detection p : preds) {
            double bestIou = 0d;
            int bestIdx = -1;
            for (int i = 0; i < gt.size(); i++) {
                if (usedGt[i]) continue;
                if (gt.get(i).getClassId() != p.getClassId()) continue;
                double iou = p.getBox().iou(gt.get(i).getBox());
                if (iou > bestIou) {
                    bestIou = iou;
                    bestIdx = i;
                }
            }
            int[] tpfp = classMap.get(p.getClassId());
            if (bestIdx >= 0 && bestIou >= iouThreshold) {
                tpfp[0]++;
                usedGt[bestIdx] = true;
            } else {
                tpfp[1]++;
            }
        }
        for (int i = 0; i < gt.size(); i++) {
            if (!usedGt[i]) classMap.get(gt.get(i).getClassId())[2]++;
        }
    }

    private List<Detection> readGroundTruth(Path labelFile, int w, int h) throws IOException {
        List<Detection> out = new ArrayList<>();
        if (!Files.exists(labelFile)) return out;
        for (String line : Files.readAllLines(labelFile)) {
            String s = line.trim();
            if (s.isEmpty()) continue;
            String[] parts = s.split("\\s+");
            if (parts.length != 5) continue;
            int cls = Integer.parseInt(parts[0]);
            float cx = Float.parseFloat(parts[1]) * w;
            float cy = Float.parseFloat(parts[2]) * h;
            float bw = Float.parseFloat(parts[3]) * w;
            float bh = Float.parseFloat(parts[4]) * h;
            BoundingBox b = BoundingBox.fromCxCyWh(cx, cy, bw, bh).clamp(w, h);
            out.add(new Detection(b, cls, mapping.labelForId(cls), 1f));
        }
        return out;
    }

    private static String stripExt(String n) {
        int dot = n.lastIndexOf('.');
        return dot < 0 ? n : n.substring(0, dot);
    }
}
