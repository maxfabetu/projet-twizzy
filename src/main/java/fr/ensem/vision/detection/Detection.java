package fr.ensem.vision.detection;

public final class Detection {

    private final BoundingBox box;
    private final int classId;
    private final String label;
    private final float confidence;
    private final long frameIndex;
    private final double timestampMs;

    public Detection(BoundingBox box, int classId, String label, float confidence) {
        this(box, classId, label, confidence, -1L, -1.0);
    }

    public Detection(BoundingBox box, int classId, String label, float confidence,
                     long frameIndex, double timestampMs) {
        this.box = box;
        this.classId = classId;
        this.label = label;
        this.confidence = confidence;
        this.frameIndex = frameIndex;
        this.timestampMs = timestampMs;
    }

    public BoundingBox getBox() { return box; }
    public int getClassId() { return classId; }
    public String getLabel() { return label; }
    public float getConfidence() { return confidence; }
    public long getFrameIndex() { return frameIndex; }
    public double getTimestampMs() { return timestampMs; }

    public Detection withFrame(long idx, double tsMs) {
        return new Detection(box, classId, label, confidence, idx, tsMs);
    }

    @Override
    public String toString() {
        return String.format("Det(class=%s conf=%.2f %s)", label, confidence, box);
    }
}
