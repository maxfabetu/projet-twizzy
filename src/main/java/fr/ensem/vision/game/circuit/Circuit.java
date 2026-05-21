package fr.ensem.vision.game.circuit;

import com.jme3.math.Vector3f;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class Circuit {

    private final List<Segment> segments;
    private final List<Vector3f> centerline;
    private final List<Vector3f> leftEdge;
    private final List<Vector3f> rightEdge;
    private final List<Float> headings;
    private final float roadWidth;
    private final float totalLength;

    public Circuit(List<Segment> segments,
                   List<Vector3f> centerline,
                   List<Vector3f> leftEdge,
                   List<Vector3f> rightEdge,
                   List<Float> headings,
                   float roadWidth,
                   float totalLength) {
        this.segments = Collections.unmodifiableList(new ArrayList<>(segments));
        this.centerline = Collections.unmodifiableList(new ArrayList<>(centerline));
        this.leftEdge = Collections.unmodifiableList(new ArrayList<>(leftEdge));
        this.rightEdge = Collections.unmodifiableList(new ArrayList<>(rightEdge));
        this.headings = Collections.unmodifiableList(new ArrayList<>(headings));
        this.roadWidth = roadWidth;
        this.totalLength = totalLength;
    }

    public List<Segment> getSegments() { return segments; }
    public List<Vector3f> getCenterline() { return centerline; }
    public List<Vector3f> getLeftEdge() { return leftEdge; }
    public List<Vector3f> getRightEdge() { return rightEdge; }
    public List<Float> getHeadings() { return headings; }
    public float getRoadWidth() { return roadWidth; }
    public float getTotalLength() { return totalLength; }

    public int sampleCount() { return centerline.size(); }

    public Vector3f sampleAt(float distance, Vector3f out) {
        if (out == null) out = new Vector3f();
        float total = totalLength;
        if (total <= 0f) { out.set(0, 0, 0); return out; }
        float t = ((distance % total) + total) % total;
        float step = total / (float) Math.max(1, centerline.size() - 1);
        int idx = (int) Math.floor(t / step);
        if (idx >= centerline.size() - 1) idx = centerline.size() - 2;
        float frac = (t - idx * step) / step;
        Vector3f a = centerline.get(idx);
        Vector3f b = centerline.get(idx + 1);
        out.set(a.x + (b.x - a.x) * frac, a.y + (b.y - a.y) * frac, a.z + (b.z - a.z) * frac);
        return out;
    }

    public float headingAt(float distance) {
        float total = totalLength;
        if (total <= 0f || headings.isEmpty()) return 0f;
        float t = ((distance % total) + total) % total;
        float step = total / (float) Math.max(1, headings.size() - 1);
        int idx = (int) Math.floor(t / step);
        if (idx >= headings.size() - 1) idx = headings.size() - 2;
        float frac = (t - idx * step) / step;
        float h0 = headings.get(idx);
        float h1 = headings.get(idx + 1);
        float diff = ((h1 - h0 + (float) Math.PI) % (2f * (float) Math.PI)) - (float) Math.PI;
        return h0 + diff * frac;
    }
}
