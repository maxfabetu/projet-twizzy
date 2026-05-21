package fr.ensem.vision.game.circuit;

public final class Segment {

    public enum Type { STRAIGHT, CURVE_LEFT, CURVE_RIGHT }

    public final Type type;
    public final float length;
    public final float curveAngleRad;
    public final float radius;

    private Segment(Type type, float length, float curveAngleRad, float radius) {
        this.type = type;
        this.length = length;
        this.curveAngleRad = curveAngleRad;
        this.radius = radius;
    }

    public static Segment straight(float length) {
        return new Segment(Type.STRAIGHT, length, 0f, 0f);
    }

    public static Segment curveLeft(float angleRad, float radius) {
        return new Segment(Type.CURVE_LEFT, angleRad * radius, angleRad, radius);
    }

    public static Segment curveRight(float angleRad, float radius) {
        return new Segment(Type.CURVE_RIGHT, angleRad * radius, angleRad, radius);
    }
}
