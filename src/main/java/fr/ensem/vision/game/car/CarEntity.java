package fr.ensem.vision.game.car;

import com.jme3.asset.AssetManager;
import com.jme3.material.Material;
import com.jme3.math.ColorRGBA;
import com.jme3.math.Quaternion;
import com.jme3.math.Vector3f;
import com.jme3.scene.Geometry;
import com.jme3.scene.Node;
import com.jme3.scene.shape.Box;
import com.jme3.scene.shape.Cylinder;
import com.jme3.scene.shape.Sphere;

public final class CarEntity {

    private final Node node;
    private final Vector3f position = new Vector3f();
    private float headingRad = 0f;
    private float speedMs = 0f;
    private float steer = 0f;

    public CarEntity(AssetManager assetManager) {
        node = new Node("twizy");
        if (assetManager == null) return;

        Material bodyMat = unlit(assetManager, new ColorRGBA(0.96f, 0.96f, 0.94f, 1f));
        Material trimMat = unlit(assetManager, new ColorRGBA(0.10f, 0.10f, 0.12f, 1f));
        Material glassMat = unlit(assetManager, new ColorRGBA(0.18f, 0.22f, 0.28f, 1f));
        Material tireMat = unlit(assetManager, new ColorRGBA(0.06f, 0.06f, 0.07f, 1f));
        Material rimMat = unlit(assetManager, new ColorRGBA(0.55f, 0.55f, 0.58f, 1f));
        Material lightMat = unlit(assetManager, new ColorRGBA(1f, 0.95f, 0.75f, 1f));
        Material rearLightMat = unlit(assetManager, new ColorRGBA(0.85f, 0.10f, 0.10f, 1f));
        Material accentMat = unlit(assetManager, new ColorRGBA(0.40f, 0.75f, 0.95f, 1f));

        Geometry chassis = box(assetManager, 0.62f, 0.18f, 1.10f, trimMat);
        chassis.setLocalTranslation(0f, 0.38f, 0f);
        node.attachChild(chassis);

        Geometry tub = box(assetManager, 0.58f, 0.34f, 1.00f, bodyMat);
        tub.setLocalTranslation(0f, 0.74f, 0.05f);
        node.attachChild(tub);

        Geometry cabin = box(assetManager, 0.56f, 0.42f, 0.72f, bodyMat);
        cabin.setLocalTranslation(0f, 1.30f, -0.05f);
        node.attachChild(cabin);

        Geometry roof = box(assetManager, 0.50f, 0.04f, 0.66f, trimMat);
        roof.setLocalTranslation(0f, 1.74f, -0.05f);
        node.attachChild(roof);

        Geometry windshield = box(assetManager, 0.52f, 0.32f, 0.04f, glassMat);
        Quaternion wsRot = new Quaternion();
        wsRot.fromAngleAxis(-0.35f, Vector3f.UNIT_X);
        windshield.setLocalRotation(wsRot);
        windshield.setLocalTranslation(0f, 1.30f, 0.70f);
        node.attachChild(windshield);

        Geometry rearWin = box(assetManager, 0.50f, 0.28f, 0.04f, glassMat);
        Quaternion rwRot = new Quaternion();
        rwRot.fromAngleAxis(0.30f, Vector3f.UNIT_X);
        rearWin.setLocalRotation(rwRot);
        rearWin.setLocalTranslation(0f, 1.32f, -0.78f);
        node.attachChild(rearWin);

        Geometry sideL = box(assetManager, 0.04f, 0.30f, 0.46f, glassMat);
        sideL.setLocalTranslation(-0.58f, 1.30f, 0.0f);
        node.attachChild(sideL);
        Geometry sideR = box(assetManager, 0.04f, 0.30f, 0.46f, glassMat);
        sideR.setLocalTranslation(0.58f, 1.30f, 0.0f);
        node.attachChild(sideR);

        Geometry frontBumper = box(assetManager, 0.62f, 0.10f, 0.10f, accentMat);
        frontBumper.setLocalTranslation(0f, 0.46f, 1.10f);
        node.attachChild(frontBumper);

        Geometry rearBumper = box(assetManager, 0.62f, 0.10f, 0.10f, trimMat);
        rearBumper.setLocalTranslation(0f, 0.46f, -1.10f);
        node.attachChild(rearBumper);

        attachLight(assetManager, lightMat, -0.40f, 0.78f, 1.05f);
        attachLight(assetManager, lightMat, 0.40f, 0.78f, 1.05f);
        attachLight(assetManager, rearLightMat, -0.46f, 0.96f, -1.10f);
        attachLight(assetManager, rearLightMat, 0.46f, 0.96f, -1.10f);

        float wheelRadius = 0.32f;
        float wheelWidth = 0.22f;
        float halfWB = 0.85f;
        float halfTrack = 0.70f;
        attachWheel(assetManager, tireMat, rimMat, wheelRadius, wheelWidth, -halfTrack, halfWB);
        attachWheel(assetManager, tireMat, rimMat, wheelRadius, wheelWidth, halfTrack, halfWB);
        attachWheel(assetManager, tireMat, rimMat, wheelRadius, wheelWidth, -halfTrack, -halfWB);
        attachWheel(assetManager, tireMat, rimMat, wheelRadius, wheelWidth, halfTrack, -halfWB);
    }

    private static Material unlit(AssetManager am, ColorRGBA color) {
        Material m = new Material(am, "Common/MatDefs/Misc/Unshaded.j3md");
        m.setColor("Color", color);
        return m;
    }

    private static Geometry box(AssetManager am, float halfX, float halfY, float halfZ, Material mat) {
        Geometry g = new Geometry("box", new Box(halfX, halfY, halfZ));
        g.setMaterial(mat);
        return g;
    }

    private void attachLight(AssetManager am, Material mat, float x, float y, float z) {
        Sphere s = new Sphere(10, 10, 0.08f);
        Geometry g = new Geometry("light", s);
        g.setMaterial(mat);
        g.setLocalTranslation(x, y, z);
        node.attachChild(g);
    }

    private void attachWheel(AssetManager am, Material tire, Material rim,
                             float radius, float width, float x, float z) {
        Cylinder tireMesh = new Cylinder(16, 20, radius, width, true);
        Geometry tireGeom = new Geometry("tire", tireMesh);
        tireGeom.setMaterial(tire);
        Quaternion rot = new Quaternion();
        rot.fromAngleAxis((float) Math.PI / 2f, Vector3f.UNIT_Y);
        tireGeom.setLocalRotation(rot);
        tireGeom.setLocalTranslation(x, radius, z);
        node.attachChild(tireGeom);

        Sphere hub = new Sphere(10, 10, radius * 0.45f);
        Geometry hubGeom = new Geometry("hub", hub);
        hubGeom.setMaterial(rim);
        hubGeom.setLocalTranslation(x, radius, z);
        node.attachChild(hubGeom);
    }

    public Node getNode() { return node; }
    public Vector3f getPosition() { return position; }
    public float getHeadingRad() { return headingRad; }
    public float getSpeedMs() { return speedMs; }
    public float getSpeedKmh() { return speedMs * 3.6f; }
    public float getSteer() { return steer; }

    public void setPosition(Vector3f p) { position.set(p); applyTransform(); }
    public void setHeading(float rad) { this.headingRad = rad; applyTransform(); }
    public void setSpeedMs(float v) { this.speedMs = v; }
    public void setSpeedKmh(float kmh) { this.speedMs = kmh / 3.6f; }
    public void setSteer(float s) { this.steer = s; }

    public void integrate(float tpf, float wheelBase) {
        if (Math.abs(speedMs) < 1e-3f && Math.abs(steer) < 1e-3f) {
            applyTransform();
            return;
        }
        float radius = (Math.abs(steer) > 1e-4f) ? wheelBase / (float) Math.tan(steer) : Float.MAX_VALUE;
        float dTheta = (Math.abs(radius) > 1e-3f && radius != Float.MAX_VALUE) ? (speedMs / radius) * tpf : 0f;
        headingRad += dTheta;
        float dx = (float) Math.sin(headingRad) * speedMs * tpf;
        float dz = (float) Math.cos(headingRad) * speedMs * tpf;
        position.x += dx;
        position.z += dz;
        applyTransform();
    }

    private void applyTransform() {
        node.setLocalTranslation(position);
        Quaternion q = new Quaternion();
        q.fromAngleAxis(headingRad, Vector3f.UNIT_Y);
        node.setLocalRotation(q);
    }
}
