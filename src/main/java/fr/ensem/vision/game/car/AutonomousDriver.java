package fr.ensem.vision.game.car;

import com.jme3.math.Vector3f;
import fr.ensem.vision.game.GameConfig;
import fr.ensem.vision.game.circuit.Circuit;

import java.util.List;

public final class AutonomousDriver {

    private final GameConfig cfg;
    private final CarEntity car;
    private float targetSpeedKmh;
    private float lastConfidence;

    public AutonomousDriver(GameConfig cfg, CarEntity car) {
        this.cfg = cfg;
        this.car = car;
        this.targetSpeedKmh = cfg.defaultSpeedKmh;
    }

    public float getTargetSpeedKmh() { return targetSpeedKmh; }
    public float getLastConfidence() { return lastConfidence; }

    public void notifySpeedLimit(int kmh, float confidence) {
        if (confidence < cfg.autonomousMinConfidence) return;
        this.targetSpeedKmh = kmh;
        this.lastConfidence = confidence;
    }

    public void update(float tpf) {
        float currentMs = car.getSpeedMs();
        float currentKmh = currentMs * 3.6f;
        float delta = targetSpeedKmh - currentKmh;
        float accelMs = cfg.autoAccel;
        float brakeMs = cfg.autoBrake;
        if (Math.abs(delta) <= cfg.autoTolerance) {
            return;
        }
        if (delta > 0f) {
            currentMs += accelMs * tpf;
        } else {
            currentMs -= brakeMs * tpf;
        }
        float maxMs = cfg.maxSpeedKmh / 3.6f;
        if (currentMs > maxMs) currentMs = maxMs;
        if (currentMs < 0f) currentMs = 0f;
        car.setSpeedMs(currentMs);
    }

    public void steerAlongCircuit(Circuit circuit) {
        if (!cfg.autoSteer || circuit == null) return;
        List<Vector3f> cl = circuit.getCenterline();
        if (cl.isEmpty()) return;
        Vector3f carPos = car.getPosition();
        int closest = 0;
        float bestDist = Float.MAX_VALUE;
        for (int i = 0; i < cl.size(); i++) {
            float d = cl.get(i).distanceSquared(carPos);
            if (d < bestDist) { bestDist = d; closest = i; }
        }
        float total = circuit.getTotalLength();
        float stepDist = total / Math.max(1f, (float) (cl.size() - 1));
        int forward = (int) Math.ceil(cfg.autoSteerLookAhead / Math.max(0.01f, stepDist));
        int targetIdx = (closest + forward) % cl.size();
        Vector3f target = cl.get(targetIdx);

        float dx = target.x - carPos.x;
        float dz = target.z - carPos.z;
        float desiredHeading = (float) Math.atan2(dx, dz);
        float currentHeading = car.getHeadingRad();
        float diff = desiredHeading - currentHeading;
        while (diff > Math.PI) diff -= 2f * Math.PI;
        while (diff < -Math.PI) diff += 2f * Math.PI;
        float steer = diff * cfg.autoSteerGain;
        if (steer > cfg.maxSteer) steer = cfg.maxSteer;
        if (steer < -cfg.maxSteer) steer = -cfg.maxSteer;
        car.setSteer(steer);
    }
}
