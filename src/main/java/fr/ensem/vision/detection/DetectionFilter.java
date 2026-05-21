package fr.ensem.vision.detection;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public final class DetectionFilter {

    private final Set<Integer> allowedClassIds;
    private final float minConfidence;

    public DetectionFilter(Set<Integer> allowedClassIds, float minConfidence) {
        this.allowedClassIds = allowedClassIds;
        this.minConfidence = minConfidence;
    }

    public List<Detection> apply(List<Detection> in) {
        List<Detection> out = new ArrayList<>(in.size());
        for (Detection d : in) {
            if (d.getConfidence() < minConfidence) continue;
            if (allowedClassIds != null && !allowedClassIds.isEmpty()
                    && !allowedClassIds.contains(d.getClassId())) continue;
            out.add(d);
        }
        return out;
    }
}
