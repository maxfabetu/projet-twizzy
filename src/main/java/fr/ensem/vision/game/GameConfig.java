package fr.ensem.vision.game;

public final class GameConfig {

    public int width = 1280;
    public int height = 720;
    public int samples = 4;
    public boolean vsync = true;

    public long seed = System.currentTimeMillis();
    public int segmentCount = 30;

    public float maxSpeedKmh = 140f;
    public float manualAcceleration = 8f;
    public float manualBrake = 14f;
    public float steerSpeed = 1.3f;
    public float steerReturn = 1.8f;
    public float maxSteer = 0.5f;
    public float defaultSpeedKmh = 50f;

    public boolean autonomous = false;
    public float autoAccel = 3.5f;
    public float autoBrake = 5.0f;
    public float autoTolerance = 1.5f;
    public float autoSteerLookAhead = 14f;
    public float autoSteerGain = 1.3f;
    public boolean autoSteer = true;

    public int detectionWidth = 640;
    public int detectionHeight = 360;
    public float detectionConfidence = 0.20f;
    public long detectionPeriodMs = 80L;
    public float autonomousMinConfidence = 0.35f;
    public float minBboxHeightPx = 60f;
    public boolean singleBestDetection = true;
    public int smoothWindow = 12;
    public int smoothResetMisses = 8;

    public boolean useCuda = true;
    public String inferenceMethod = "yolo";

    public float roadWidth = 8f;
    public float straightLength = 60f;
    public float curveRadius = 35f;
    public float curveAngleDegMin = 25f;
    public float curveAngleDegMax = 55f;
    public float signMinDistance = 80f;
    public float signMaxDistance = 150f;
    public float signLateralOffset = 4.5f;
    public float signHeight = 3.0f;
    public float signSize = 2.8f;
    public float cameraFovDeg = 35f;

    public float cameraHeight = 1.7f;
    public float cameraBack = 6f;
    public float cameraLookAhead = 8f;
}
