package fr.ensem.vision.game;

import com.jme3.app.SimpleApplication;
import com.jme3.font.BitmapFont;
import com.jme3.input.KeyInput;
import com.jme3.input.controls.ActionListener;
import com.jme3.input.controls.KeyTrigger;
import com.jme3.math.Vector3f;
import com.jme3.scene.Node;
import com.jme3.system.AppSettings;
import fr.ensem.vision.config.AppConfig;
import fr.ensem.vision.config.ClassMapping;
import fr.ensem.vision.detection.Detection;
import fr.ensem.vision.game.car.AutonomousDriver;
import fr.ensem.vision.game.car.CarController;
import fr.ensem.vision.game.car.CarEntity;
import fr.ensem.vision.game.circuit.Circuit;
import fr.ensem.vision.game.circuit.CircuitGenerator;
import fr.ensem.vision.game.circuit.CircuitMeshBuilder;
import fr.ensem.vision.game.detection.DetectionOverlay;
import fr.ensem.vision.game.detection.DetectionService;
import fr.ensem.vision.game.detection.DetectionSmoother;
import fr.ensem.vision.game.env.EnvironmentBuilder;
import fr.ensem.vision.game.hud.GameHud;
import fr.ensem.vision.game.sign.SignPlacer;
import fr.ensem.vision.game.sign.SignTextureLoader;
import fr.ensem.vision.inference.InferenceEngine;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Random;

public final class GameApp extends SimpleApplication {

    private static final Logger LOG = LoggerFactory.getLogger(GameApp.class);

    private final GameConfig cfg;
    private final InferenceEngine engine;

    private Circuit circuit;
    private CarEntity car;
    private CarController controller;
    private AutonomousDriver autonomous;
    private SignPlacer signPlacer;
    private SignTextureLoader signTextures;
    private CircuitMeshBuilder meshBuilder;
    private Node circuitNode;
    private Node signsNode;

    private DetectionService detection;
    private DetectionOverlay overlay;
    private DetectionSmoother smoother;
    private GameHud hud;

    private float wheelBase = 2.6f;
    private long lastInferenceCount = -1L;
    private Detection lastSmoothed;
    private long lastFpsLog = 0L;

    public GameApp(GameConfig cfg, InferenceEngine engine) {
        super();
        this.cfg = cfg;
        this.engine = engine;
        AppSettings s = new AppSettings(true);
        s.setTitle("Vision ENSEM - Driving Sim");
        s.setResolution(cfg.width, cfg.height);
        s.setSamples(cfg.samples);
        s.setVSync(cfg.vsync);
        s.setGammaCorrection(true);
        setSettings(s);
        setShowSettings(false);
        setDisplayStatView(false);
        setDisplayFps(false);
    }

    @Override
    public void simpleInitApp() {
        flyCam.setEnabled(false);
        viewPort.setBackgroundColor(new com.jme3.math.ColorRGBA(0.55f, 0.72f, 0.92f, 1f));
        float aspect = (float) cam.getWidth() / Math.max(1f, (float) cam.getHeight());
        cam.setFrustumPerspective(cfg.cameraFovDeg, aspect, 0.5f, 2000f);
        DetectionService.setRenderer(renderer);

        EnvironmentBuilder env = new EnvironmentBuilder(assetManager);
        env.addSky(rootNode);
        env.addLights(rootNode);

        signTextures = new SignTextureLoader(assetManager);
        signPlacer = new SignPlacer(assetManager, signTextures, cfg);
        meshBuilder = new CircuitMeshBuilder(assetManager);

        car = new CarEntity(assetManager);
        rootNode.attachChild(car.getNode());
        controller = new CarController(cfg, car);
        controller.registerInputs(inputManager);
        controller.setOnRegen(this::regenerateCircuit);
        autonomous = new AutonomousDriver(cfg, car);

        BitmapFont defaultFont = guiFont != null ? guiFont
                : assetManager.loadFont("Interface/Fonts/Default.fnt");
        hud = new GameHud(guiNode, defaultFont);
        hud.layout(cfg.width, cfg.height);
        overlay = new DetectionOverlay(assetManager, defaultFont, guiNode);
        overlay.configure(cfg.width, cfg.height, cfg.width, cfg.height);

        ClassMapping mapping = AppConfig.load().classMapping();
        smoother = new DetectionSmoother(mapping, cfg.smoothWindow, cfg.smoothResetMisses);

        detection = new DetectionService(engine, cfg.detectionPeriodMs);
        viewPort.addProcessor(detection);

        inputManager.addMapping("quit", new KeyTrigger(KeyInput.KEY_ESCAPE));
        inputManager.addListener((ActionListener) (n, p, t) -> { if ("quit".equals(n) && p) stop(); }, "quit");

        regenerateCircuit();
        LOG.info("Game started, seed={} segments={} autonomous={}",
                cfg.seed, cfg.segmentCount, cfg.autonomous);
    }

    private void regenerateCircuit() {
        if (circuitNode != null) rootNode.detachChild(circuitNode);
        if (signsNode != null) rootNode.detachChild(signsNode);
        long seed = (circuit == null) ? cfg.seed : new Random().nextLong();
        cfg.seed = seed;
        CircuitGenerator gen = new CircuitGenerator(cfg);
        circuit = gen.generate(seed);
        circuitNode = meshBuilder.build(circuit);
        rootNode.attachChild(circuitNode);
        signsNode = new Node("signs");
        rootNode.attachChild(signsNode);
        signPlacer.place(circuit, seed, signsNode);
        Vector3f startPos = new Vector3f();
        circuit.sampleAt(0f, startPos);
        car.setPosition(startPos);
        car.setHeading(circuit.headingAt(0f));
        car.setSpeedMs(0f);
        car.setSteer(0f);
        if (smoother != null) smoother.reset();
    }

    @Override
    public void simpleUpdate(float tpf) {
        if (tpf > 0.1f) tpf = 0.1f;
        controller.update(tpf);
        if (controller.isAutonomous()) {
            autonomous.steerAlongCircuit(circuit);
            autonomous.update(tpf);
        }
        car.integrate(tpf, wheelBase);

        Vector3f camPos = computeCameraPosition();
        Vector3f camTarget = computeCameraTarget();
        cam.setLocation(camPos);
        cam.lookAt(camTarget, Vector3f.UNIT_Y);

        List<Detection> rawDets = detection.getLatestDetections();
        List<Detection> filtered = new java.util.ArrayList<>(rawDets.size());
        for (Detection d : rawDets) {
            if (d.getBox().getHeight() < cfg.minBboxHeightPx) continue;
            filtered.add(d);
        }
        if (cfg.singleBestDetection && filtered.size() > 1) {
            Detection best = filtered.get(0);
            for (Detection d : filtered) {
                if (d.getBox().area() > best.getBox().area()) best = d;
            }
            filtered = java.util.Collections.singletonList(best);
        }
        long currentInferenceCount = detection.getDetectionCount();
        if (currentInferenceCount != lastInferenceCount) {
            lastInferenceCount = currentInferenceCount;
            Detection rawSingle = filtered.isEmpty() ? null : filtered.get(0);
            lastSmoothed = smoother.smooth(rawSingle);
        }
        if (lastSmoothed != null && !filtered.isEmpty()) {
            Detection withCurrentBox = new Detection(
                    filtered.get(0).getBox(),
                    lastSmoothed.getClassId(),
                    lastSmoothed.getLabel(),
                    lastSmoothed.getConfidence());
            filtered = java.util.Collections.singletonList(withCurrentBox);
        } else if (lastSmoothed != null) {
            filtered = java.util.Collections.singletonList(lastSmoothed);
        } else {
            filtered = java.util.Collections.emptyList();
        }
        for (Detection d : filtered) {
            if (d.getConfidence() < cfg.autonomousMinConfidence) continue;
            try {
                int kmh = Integer.parseInt(d.getLabel());
                autonomous.notifySpeedLimit(kmh, d.getConfidence());
            } catch (NumberFormatException ignored) {}
        }
        overlay.update(filtered);
        String lastClass = null;
        float lastConf = 0f;
        for (Detection d : filtered) {
            if (d.getConfidence() > lastConf) { lastConf = d.getConfidence(); lastClass = d.getLabel(); }
        }
        float renderFps = 1f / Math.max(tpf, 1e-3f);
        hud.update(car.getSpeedKmh(), autonomous.getTargetSpeedKmh(), controller.isAutonomous(),
                renderFps, detection.getDetectionFps(), lastClass, lastConf);

        long now = System.currentTimeMillis();
        if (now - lastFpsLog > 4000) {
            LOG.info("speed={}km/h target={}km/h dets={} renderFps={} detectFps={}",
                    Math.round(car.getSpeedKmh()), Math.round(autonomous.getTargetSpeedKmh()),
                    filtered.size(), Math.round(renderFps), Math.round(detection.getDetectionFps()));
            lastFpsLog = now;
        }
    }

    private Vector3f computeCameraPosition() {
        float heading = car.getHeadingRad();
        Vector3f pos = car.getPosition();
        float dx = -(float) Math.sin(heading) * cfg.cameraBack;
        float dz = -(float) Math.cos(heading) * cfg.cameraBack;
        return new Vector3f(pos.x + dx, pos.y + cfg.cameraHeight, pos.z + dz);
    }

    private Vector3f computeCameraTarget() {
        float heading = car.getHeadingRad();
        Vector3f pos = car.getPosition();
        float dx = (float) Math.sin(heading) * cfg.cameraLookAhead;
        float dz = (float) Math.cos(heading) * cfg.cameraLookAhead;
        return new Vector3f(pos.x + dx, pos.y + 1.2f, pos.z + dz);
    }

    @Override
    public void destroy() {
        if (detection != null) detection.cleanup();
        super.destroy();
    }
}
