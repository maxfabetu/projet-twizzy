package fr.ensem.vision.game.car;

import com.jme3.input.InputManager;
import com.jme3.input.KeyInput;
import com.jme3.input.controls.ActionListener;
import com.jme3.input.controls.AnalogListener;
import com.jme3.input.controls.KeyTrigger;
import fr.ensem.vision.game.GameConfig;

public final class CarController implements AnalogListener, ActionListener {

    public static final String ACCEL = "carAccel";
    public static final String BRAKE = "carBrake";
    public static final String LEFT = "carLeft";
    public static final String RIGHT = "carRight";
    public static final String HARD_BRAKE = "carHardBrake";
    public static final String TOGGLE_AUTO = "toggleAuto";
    public static final String REGEN = "regen";

    private final GameConfig cfg;
    private final CarEntity car;

    private boolean accelHeld;
    private boolean brakeHeld;
    private boolean leftHeld;
    private boolean rightHeld;
    private boolean hardBrakeHeld;
    private boolean autonomous;

    private Runnable onToggleAuto;
    private Runnable onRegen;

    public CarController(GameConfig cfg, CarEntity car) {
        this.cfg = cfg;
        this.car = car;
        this.autonomous = cfg.autonomous;
    }

    public void registerInputs(InputManager im) {
        im.addMapping(ACCEL, new KeyTrigger(KeyInput.KEY_Z), new KeyTrigger(KeyInput.KEY_W), new KeyTrigger(KeyInput.KEY_UP));
        im.addMapping(BRAKE, new KeyTrigger(KeyInput.KEY_S), new KeyTrigger(KeyInput.KEY_DOWN));
        im.addMapping(LEFT, new KeyTrigger(KeyInput.KEY_Q), new KeyTrigger(KeyInput.KEY_A), new KeyTrigger(KeyInput.KEY_LEFT));
        im.addMapping(RIGHT, new KeyTrigger(KeyInput.KEY_D), new KeyTrigger(KeyInput.KEY_RIGHT));
        im.addMapping(HARD_BRAKE, new KeyTrigger(KeyInput.KEY_SPACE));
        im.addMapping(TOGGLE_AUTO, new KeyTrigger(KeyInput.KEY_TAB));
        im.addMapping(REGEN, new KeyTrigger(KeyInput.KEY_R));

        im.addListener(this, ACCEL, BRAKE, LEFT, RIGHT, HARD_BRAKE);
        im.addListener(this, TOGGLE_AUTO, REGEN);
    }

    public void setOnToggleAuto(Runnable r) { this.onToggleAuto = r; }
    public void setOnRegen(Runnable r) { this.onRegen = r; }
    public boolean isAutonomous() { return autonomous; }
    public void setAutonomous(boolean v) { this.autonomous = v; }

    public void update(float tpf) {
        float steerTarget = 0f;
        if (leftHeld) steerTarget += cfg.maxSteer;
        if (rightHeld) steerTarget -= cfg.maxSteer;
        float steer = car.getSteer();
        if (steerTarget != 0f) {
            steer += (steerTarget - steer) * Math.min(1f, cfg.steerSpeed * tpf);
        } else {
            steer -= Math.signum(steer) * Math.min(Math.abs(steer), cfg.steerReturn * tpf);
        }
        steer = Math.max(-cfg.maxSteer, Math.min(cfg.maxSteer, steer));
        car.setSteer(steer);

        if (!autonomous) {
            float speed = car.getSpeedMs();
            float maxMs = cfg.maxSpeedKmh / 3.6f;
            if (hardBrakeHeld) {
                float dec = cfg.manualBrake * 2f * tpf;
                speed -= Math.signum(speed) * Math.min(Math.abs(speed), dec);
            } else {
                if (accelHeld) speed += cfg.manualAcceleration * tpf;
                if (brakeHeld) speed -= cfg.manualBrake * tpf;
                if (!accelHeld && !brakeHeld) {
                    float dec = 1.0f * tpf;
                    speed -= Math.signum(speed) * Math.min(Math.abs(speed), dec);
                }
            }
            if (speed > maxMs) speed = maxMs;
            if (speed < -maxMs * 0.4f) speed = -maxMs * 0.4f;
            car.setSpeedMs(speed);
        }
    }

    @Override
    public void onAnalog(String name, float value, float tpf) {
    }

    @Override
    public void onAction(String name, boolean isPressed, float tpf) {
        switch (name) {
            case ACCEL: accelHeld = isPressed; break;
            case BRAKE: brakeHeld = isPressed; break;
            case LEFT: leftHeld = isPressed; break;
            case RIGHT: rightHeld = isPressed; break;
            case HARD_BRAKE: hardBrakeHeld = isPressed; break;
            case TOGGLE_AUTO:
                if (isPressed) {
                    autonomous = !autonomous;
                    if (onToggleAuto != null) onToggleAuto.run();
                }
                break;
            case REGEN:
                if (isPressed && onRegen != null) onRegen.run();
                break;
            default:
                break;
        }
    }
}
