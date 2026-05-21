package fr.ensem.vision.game.circuit;

import com.jme3.math.Vector3f;
import fr.ensem.vision.game.GameConfig;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public final class CircuitGenerator {

    private final GameConfig cfg;

    public CircuitGenerator(GameConfig cfg) {
        this.cfg = cfg;
    }

    public Circuit generate(long seed) {
        Random rng = new Random(seed);
        List<Segment> segments = new ArrayList<>(cfg.segmentCount);
        int sinceCurve = 0;
        for (int i = 0; i < cfg.segmentCount; i++) {
            float roll = rng.nextFloat();
            if (i == 0 || sinceCurve < 1) {
                segments.add(Segment.straight(cfg.straightLength * (0.8f + rng.nextFloat() * 0.4f)));
                sinceCurve++;
                continue;
            }
            if (roll < 0.55f) {
                segments.add(Segment.straight(cfg.straightLength * (0.7f + rng.nextFloat() * 0.6f)));
                sinceCurve++;
            } else {
                float angleDeg = cfg.curveAngleDegMin + rng.nextFloat() * (cfg.curveAngleDegMax - cfg.curveAngleDegMin);
                float angleRad = (float) Math.toRadians(angleDeg);
                if (roll < 0.775f) segments.add(Segment.curveLeft(angleRad, cfg.curveRadius));
                else segments.add(Segment.curveRight(angleRad, cfg.curveRadius));
                sinceCurve = 0;
            }
        }
        return build(segments);
    }

    private Circuit build(List<Segment> segments) {
        List<Vector3f> center = new ArrayList<>();
        List<Vector3f> left = new ArrayList<>();
        List<Vector3f> right = new ArrayList<>();
        List<Float> headings = new ArrayList<>();

        Vector3f pos = new Vector3f(0f, 0f, 0f);
        float heading = 0f;
        appendSample(pos, heading, center, left, right, headings);

        float resolution = 2f;
        float total = 0f;
        for (Segment s : segments) {
            int steps = Math.max(1, (int) Math.ceil(s.length / resolution));
            float stepLen = s.length / steps;
            switch (s.type) {
                case STRAIGHT:
                    for (int i = 0; i < steps; i++) {
                        pos = move(pos, heading, stepLen);
                        total += stepLen;
                        appendSample(pos, heading, center, left, right, headings);
                    }
                    break;
                case CURVE_LEFT:
                case CURVE_RIGHT: {
                    float dirSign = (s.type == Segment.Type.CURVE_LEFT) ? +1f : -1f;
                    float angleStep = s.curveAngleRad / steps * dirSign;
                    for (int i = 0; i < steps; i++) {
                        float midHeading = heading + angleStep * 0.5f;
                        pos = move(pos, midHeading, stepLen);
                        heading += angleStep;
                        total += stepLen;
                        appendSample(pos, heading, center, left, right, headings);
                    }
                    break;
                }
            }
        }
        return new Circuit(segments, center, left, right, headings, cfg.roadWidth, total);
    }

    private Vector3f move(Vector3f pos, float heading, float dist) {
        float dx = (float) Math.sin(heading) * dist;
        float dz = (float) Math.cos(heading) * dist;
        return new Vector3f(pos.x + dx, pos.y, pos.z + dz);
    }

    private void appendSample(Vector3f pos, float heading,
                              List<Vector3f> center, List<Vector3f> left, List<Vector3f> right,
                              List<Float> headings) {
        float half = cfg.roadWidth * 0.5f;
        float nx = (float) Math.cos(heading);
        float nz = -(float) Math.sin(heading);
        center.add(pos.clone());
        left.add(new Vector3f(pos.x - nx * half, pos.y, pos.z - nz * half));
        right.add(new Vector3f(pos.x + nx * half, pos.y, pos.z + nz * half));
        headings.add(heading);
    }
}
