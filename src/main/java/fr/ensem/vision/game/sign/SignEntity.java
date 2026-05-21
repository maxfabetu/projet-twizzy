package fr.ensem.vision.game.sign;

import com.jme3.asset.AssetManager;
import com.jme3.material.Material;
import com.jme3.material.RenderState;
import com.jme3.math.ColorRGBA;
import com.jme3.math.Quaternion;
import com.jme3.math.Vector3f;
import com.jme3.renderer.queue.RenderQueue;
import com.jme3.scene.Geometry;
import com.jme3.scene.Node;
import com.jme3.scene.shape.Cylinder;
import com.jme3.scene.shape.Quad;
import com.jme3.texture.Texture;

public final class SignEntity {

    private final Node node;
    private final String shortName;
    private final int speedKmh;
    private final Vector3f worldPosition;

    public SignEntity(AssetManager assetManager, SignTextureLoader textures,
                      String shortName, Vector3f position, float yawRad,
                      float signSize, float poleHeight) {
        this.shortName = shortName;
        this.speedKmh = Integer.parseInt(shortName);
        this.worldPosition = position.clone();
        node = new Node("sign-" + shortName);

        Cylinder pole = new Cylinder(8, 12, 0.07f, poleHeight, true);
        Geometry poleGeom = new Geometry("pole", pole);
        Material mp = new Material(assetManager, "Common/MatDefs/Misc/Unshaded.j3md");
        mp.setColor("Color", new ColorRGBA(0.7f, 0.7f, 0.72f, 1f));
        poleGeom.setMaterial(mp);
        Quaternion poleRot = new Quaternion();
        poleRot.fromAngleAxis((float) Math.PI / 2f, Vector3f.UNIT_X);
        poleGeom.setLocalRotation(poleRot);
        poleGeom.setLocalTranslation(0f, poleHeight * 0.5f, 0f);
        node.attachChild(poleGeom);

        Quad q = new Quad(signSize, signSize, true);
        Geometry plate = new Geometry("plate-" + shortName, q);
        Material mat = new Material(assetManager, "Common/MatDefs/Misc/Unshaded.j3md");
        Texture tex = textures.get(shortName);
        mat.setTexture("ColorMap", tex);
        mat.setTransparent(true);
        mat.getAdditionalRenderState().setBlendMode(RenderState.BlendMode.Alpha);
        mat.getAdditionalRenderState().setFaceCullMode(RenderState.FaceCullMode.Off);
        plate.setQueueBucket(RenderQueue.Bucket.Transparent);
        plate.setMaterial(mat);
        plate.setLocalTranslation(-signSize * 0.5f, poleHeight - signSize * 0.05f, 0.04f);
        node.attachChild(plate);

        Quaternion yaw = new Quaternion();
        yaw.fromAngleAxis(yawRad, Vector3f.UNIT_Y);
        node.setLocalRotation(yaw);
        node.setLocalTranslation(position);
    }

    public Node getNode() { return node; }
    public String getShortName() { return shortName; }
    public int getSpeedKmh() { return speedKmh; }
    public Vector3f getWorldPosition() { return worldPosition; }
}
