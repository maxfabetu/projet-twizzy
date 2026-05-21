package fr.ensem.vision.game.detection;

import fr.ensem.vision.config.ClassMapping;
import fr.ensem.vision.detection.Detection;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.Map;

public final class DetectionSmoother {

    private final ClassMapping mapping;
    private final int windowSize;
    private final Deque<Integer> classes = new ArrayDeque<>();
    private final Deque<Float> confidences = new ArrayDeque<>();
    private int missCount = 0;
    private final int resetAfterMisses;

    public DetectionSmoother(ClassMapping mapping, int windowSize, int resetAfterMisses) {
        this.mapping = mapping;
        this.windowSize = Math.max(2, windowSize);
        this.resetAfterMisses = Math.max(1, resetAfterMisses);
    }

    public Detection smooth(Detection raw) {
        if (raw == null) {
            missCount++;
            if (missCount >= resetAfterMisses) {
                classes.clear();
                confidences.clear();
                missCount = 0;
            }
            return null;
        }
        missCount = 0;
        if (classes.size() >= windowSize) {
            classes.pollFirst();
            confidences.pollFirst();
        }
        classes.addLast(raw.getClassId());
        confidences.addLast(raw.getConfidence());

        Map<Integer, Integer> counts = new HashMap<>();
        Map<Integer, Float> confSum = new HashMap<>();
        Integer[] cls = classes.toArray(new Integer[0]);
        Float[] cnf = confidences.toArray(new Float[0]);
        for (int i = 0; i < cls.length; i++) {
            int c = cls[i];
            counts.merge(c, 1, Integer::sum);
            confSum.merge(c, cnf[i], Float::sum);
        }
        int bestId = raw.getClassId();
        int bestCount = -1;
        for (Map.Entry<Integer, Integer> e : counts.entrySet()) {
            if (e.getValue() > bestCount) {
                bestCount = e.getValue();
                bestId = e.getKey();
            } else if (e.getValue() == bestCount) {
                if (confSum.get(e.getKey()) > confSum.getOrDefault(bestId, 0f)) {
                    bestId = e.getKey();
                }
            }
        }
        float avgConf = confSum.get(bestId) / counts.get(bestId);
        String label = mapping.labelForId(bestId);
        return new Detection(raw.getBox(), bestId, label, avgConf);
    }

    public int bufferSize() { return classes.size(); }

    public void reset() {
        classes.clear();
        confidences.clear();
        missCount = 0;
    }
}
