package fr.ensem.vision.inference.yolo;

import fr.ensem.vision.detection.Detection;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class NonMaxSuppression {

    private final float iouThreshold;

    public NonMaxSuppression(float iouThreshold) {
        this.iouThreshold = iouThreshold;
    }

    public List<Detection> apply(List<Detection> input) {
        Map<Integer, List<Detection>> byClass = new HashMap<>();
        for (Detection d : input) byClass.computeIfAbsent(d.getClassId(), k -> new ArrayList<>()).add(d);

        List<Detection> kept = new ArrayList<>();
        for (Map.Entry<Integer, List<Detection>> e : byClass.entrySet()) {
            kept.addAll(suppressPerClass(e.getValue()));
        }
        kept.sort(Comparator.comparingDouble((Detection d) -> -d.getConfidence()));
        return kept;
    }

    private List<Detection> suppressPerClass(List<Detection> dets) {
        dets.sort(Comparator.comparingDouble((Detection d) -> -d.getConfidence()));
        List<Detection> kept = new ArrayList<>();
        boolean[] suppressed = new boolean[dets.size()];
        for (int i = 0; i < dets.size(); i++) {
            if (suppressed[i]) continue;
            Detection a = dets.get(i);
            kept.add(a);
            for (int j = i + 1; j < dets.size(); j++) {
                if (suppressed[j]) continue;
                Detection b = dets.get(j);
                if (a.getBox().iou(b.getBox()) > iouThreshold) suppressed[j] = true;
            }
        }
        return kept;
    }
}
