package fr.ensem.vision.game.car;

import fr.ensem.vision.game.GameConfig;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AutonomousDriverTest {

    @Test
    void ignoresLowConfidenceNotification() {
        GameConfig cfg = new GameConfig();
        cfg.autonomousMinConfidence = 0.5f;
        CarEntity car = newCar();
        AutonomousDriver d = new AutonomousDriver(cfg, car);
        d.notifySpeedLimit(90, 0.3f);
        assertEquals(cfg.defaultSpeedKmh, d.getTargetSpeedKmh(), 1e-3f);
    }

    @Test
    void updatesTargetAfterValidDetection() {
        GameConfig cfg = new GameConfig();
        cfg.autonomousMinConfidence = 0.5f;
        AutonomousDriver d = new AutonomousDriver(cfg, newCar());
        d.notifySpeedLimit(110, 0.7f);
        assertEquals(110f, d.getTargetSpeedKmh(), 1e-3f);
    }

    @Test
    void acceleratesTowardsTarget() {
        GameConfig cfg = new GameConfig();
        cfg.autoAccel = 3f;
        cfg.autoBrake = 5f;
        cfg.autoTolerance = 0.5f;
        cfg.autonomousMinConfidence = 0.5f;
        CarEntity car = newCar();
        car.setSpeedKmh(10f);
        AutonomousDriver d = new AutonomousDriver(cfg, car);
        d.notifySpeedLimit(50, 0.8f);
        for (int i = 0; i < 200; i++) d.update(0.05f);
        assertTrue(car.getSpeedKmh() > 40f);
    }

    @Test
    void brakesTowardsLowerLimit() {
        GameConfig cfg = new GameConfig();
        cfg.autoAccel = 3f;
        cfg.autoBrake = 5f;
        cfg.autoTolerance = 0.5f;
        cfg.autonomousMinConfidence = 0.5f;
        CarEntity car = newCar();
        car.setSpeedKmh(120f);
        AutonomousDriver d = new AutonomousDriver(cfg, car);
        d.notifySpeedLimit(50, 0.8f);
        for (int i = 0; i < 200; i++) d.update(0.05f);
        assertTrue(car.getSpeedKmh() < 60f);
    }

    private CarEntity newCar() {
        return new CarEntity(null);
    }
}
