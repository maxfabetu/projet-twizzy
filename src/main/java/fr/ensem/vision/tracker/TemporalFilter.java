package fr.ensem.vision.tracker;

import fr.ensem.vision.detection.Detection;

import java.util.List;

public final class TemporalFilter {

    private final int minDetections;
    private final float minMeanConfidence;

    public TemporalFilter(int minDetections, float minMeanConfidence) {
        this.minDetections = Math.max(1, minDetections);
        this.minMeanConfidence = minMeanConfidence;
    }

    public boolean isConfirmed(List<Detection> detections) {
        if (detections.size() < minDetections) return false;
        float sum = 0f;
        for (Detection d : detections) sum += d.getConfidence();
        float mean = sum / detections.size();
        return mean >= minMeanConfidence;
    }

    public int minDetections() { return minDetections; }
    public float minMeanConfidence() { return minMeanConfidence; }
}
