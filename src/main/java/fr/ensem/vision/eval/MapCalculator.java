package fr.ensem.vision.eval;

import fr.ensem.vision.detection.BoundingBox;
import fr.ensem.vision.detection.Detection;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class MapCalculator {

    public static final class ImageGt {
        public final List<Detection> ground;
        public final int width;
        public final int height;

        public ImageGt(List<Detection> ground, int width, int height) {
            this.ground = ground;
            this.width = width;
            this.height = height;
        }
    }

    public static final class Sample {
        public final ImageGt gt;
        public final List<Detection> predictions;

        public Sample(ImageGt gt, List<Detection> predictions) {
            this.gt = gt;
            this.predictions = predictions;
        }
    }

    private final int numClasses;

    public MapCalculator(int numClasses) {
        this.numClasses = numClasses;
    }

    public Map<Integer, Double> averagePrecisionPerClass(List<Sample> samples, double iouThreshold) {
        Map<Integer, Double> out = new LinkedHashMap<>();
        for (int c = 0; c < numClasses; c++) {
            out.put(c, ap(samples, c, iouThreshold));
        }
        return out;
    }

    public double meanAveragePrecision(List<Sample> samples, double iouThreshold) {
        Map<Integer, Double> per = averagePrecisionPerClass(samples, iouThreshold);
        double sum = 0d;
        for (double v : per.values()) sum += v;
        return per.isEmpty() ? 0d : sum / per.size();
    }

    public double mapRange(List<Sample> samples, double start, double end, double step) {
        double sum = 0d;
        int count = 0;
        for (double t = start; t <= end + 1e-9; t += step) {
            sum += meanAveragePrecision(samples, t);
            count++;
        }
        return count == 0 ? 0d : sum / count;
    }

    private double ap(List<Sample> samples, int classId, double iouThreshold) {
        List<double[]> entries = new ArrayList<>();
        int totalGt = 0;
        for (int s = 0; s < samples.size(); s++) {
            Sample sm = samples.get(s);
            for (Detection p : sm.predictions) {
                if (p.getClassId() != classId) continue;
                entries.add(new double[] { s, p.getConfidence(), p.getBox().getX(), p.getBox().getY(),
                        p.getBox().getX2(), p.getBox().getY2() });
            }
            for (Detection g : sm.gt.ground) if (g.getClassId() == classId) totalGt++;
        }
        if (totalGt == 0 || entries.isEmpty()) return 0d;

        entries.sort(Comparator.comparingDouble((double[] e) -> -e[1]));
        Map<Integer, boolean[]> matched = new HashMap<>();

        int tp = 0;
        int fp = 0;
        List<double[]> pr = new ArrayList<>();
        for (double[] e : entries) {
            int sIdx = (int) e[0];
            Sample sm = samples.get(sIdx);
            boolean[] used = matched.computeIfAbsent(sIdx, k -> new boolean[sm.gt.ground.size()]);
            BoundingBox pb = BoundingBox.fromXyxy((float) e[2], (float) e[3], (float) e[4], (float) e[5]);
            double bestIou = 0d;
            int bestIdx = -1;
            for (int i = 0; i < sm.gt.ground.size(); i++) {
                Detection g = sm.gt.ground.get(i);
                if (g.getClassId() != classId) continue;
                if (used[i]) continue;
                double iou = pb.iou(g.getBox());
                if (iou > bestIou) {
                    bestIou = iou;
                    bestIdx = i;
                }
            }
            if (bestIdx >= 0 && bestIou >= iouThreshold) {
                used[bestIdx] = true;
                tp++;
            } else {
                fp++;
            }
            double recall = tp / (double) totalGt;
            double precision = tp / (double) (tp + fp);
            pr.add(new double[] { recall, precision });
        }

        double ap = 0d;
        double prevR = 0d;
        double maxP = 0d;
        for (int i = pr.size() - 1; i >= 0; i--) {
            if (pr.get(i)[1] > maxP) maxP = pr.get(i)[1];
            pr.get(i)[1] = maxP;
        }
        for (double[] p : pr) {
            ap += (p[0] - prevR) * p[1];
            prevR = p[0];
        }
        return ap;
    }
}
