package fr.ensem.vision.tracker;

import fr.ensem.vision.detection.Detection;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

public final class DetectionAggregator {

    public static final class FrameBucket {
        public final long frameIndex;
        public final List<Detection> detections;

        public FrameBucket(long frameIndex, List<Detection> detections) {
            this.frameIndex = frameIndex;
            this.detections = detections;
        }
    }

    private final int windowSize;
    private final Deque<FrameBucket> window;

    public DetectionAggregator(int windowSize) {
        this.windowSize = Math.max(1, windowSize);
        this.window = new ArrayDeque<>(this.windowSize);
    }

    public void add(long frameIndex, List<Detection> detections) {
        if (window.size() == windowSize) window.pollFirst();
        window.addLast(new FrameBucket(frameIndex, new ArrayList<>(detections)));
    }

    public List<Detection> classDetections(int classId) {
        List<Detection> out = new ArrayList<>();
        for (FrameBucket b : window) {
            for (Detection d : b.detections) {
                if (d.getClassId() == classId) out.add(d);
            }
        }
        return out;
    }

    public int windowSize() { return windowSize; }
    public int frames() { return window.size(); }

    public void clear() { window.clear(); }
}
