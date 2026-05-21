package fr.ensem.vision.detection;

public final class BoundingBox {

    private final float x;
    private final float y;
    private final float width;
    private final float height;

    public BoundingBox(float x, float y, float width, float height) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }

    public static BoundingBox fromXyxy(float x1, float y1, float x2, float y2) {
        return new BoundingBox(x1, y1, x2 - x1, y2 - y1);
    }

    public static BoundingBox fromCxCyWh(float cx, float cy, float w, float h) {
        return new BoundingBox(cx - w / 2f, cy - h / 2f, w, h);
    }

    public float getX() { return x; }
    public float getY() { return y; }
    public float getWidth() { return width; }
    public float getHeight() { return height; }
    public float getX2() { return x + width; }
    public float getY2() { return y + height; }
    public float getCenterX() { return x + width / 2f; }
    public float getCenterY() { return y + height / 2f; }
    public float area() { return Math.max(0f, width) * Math.max(0f, height); }

    public float iou(BoundingBox other) {
        float ix1 = Math.max(this.x, other.x);
        float iy1 = Math.max(this.y, other.y);
        float ix2 = Math.min(this.getX2(), other.getX2());
        float iy2 = Math.min(this.getY2(), other.getY2());
        float iw = Math.max(0f, ix2 - ix1);
        float ih = Math.max(0f, iy2 - iy1);
        float inter = iw * ih;
        float union = this.area() + other.area() - inter;
        return union <= 0f ? 0f : inter / union;
    }

    public BoundingBox clamp(float maxX, float maxY) {
        float nx = Math.max(0f, Math.min(x, maxX));
        float ny = Math.max(0f, Math.min(y, maxY));
        float nx2 = Math.max(0f, Math.min(getX2(), maxX));
        float ny2 = Math.max(0f, Math.min(getY2(), maxY));
        return new BoundingBox(nx, ny, nx2 - nx, ny2 - ny);
    }

    public BoundingBox normalized(float imgW, float imgH) {
        return new BoundingBox(x / imgW, y / imgH, width / imgW, height / imgH);
    }

    @Override
    public String toString() {
        return String.format("BBox(x=%.1f y=%.1f w=%.1f h=%.1f)", x, y, width, height);
    }
}
