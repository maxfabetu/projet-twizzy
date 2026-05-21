package fr.ensem.vision.game.detection;

import com.jme3.asset.AssetManager;
import com.jme3.font.BitmapFont;
import com.jme3.font.BitmapText;
import com.jme3.material.Material;
import com.jme3.material.RenderState;
import com.jme3.math.ColorRGBA;
import com.jme3.math.Vector3f;
import com.jme3.scene.Geometry;
import com.jme3.scene.Node;
import com.jme3.scene.shape.Quad;
import fr.ensem.vision.detection.BoundingBox;
import fr.ensem.vision.detection.Detection;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class DetectionOverlay {

    private final AssetManager assetManager;
    private final BitmapFont font;
    private final Node hud;
    private final Map<Integer, ColorRGBA> classColors = new HashMap<>();

    private final Node overlayNode;
    private int screenWidth;
    private int screenHeight;
    private int sourceWidth;
    private int sourceHeight;

    public DetectionOverlay(AssetManager assetManager, BitmapFont font, Node hud) {
        this.assetManager = assetManager;
        this.font = font;
        this.hud = hud;
        this.overlayNode = new Node("detectionOverlay");
        hud.attachChild(overlayNode);
        classColors.put(0, new ColorRGBA(0.16f, 0.86f, 0.16f, 1f));
        classColors.put(1, new ColorRGBA(0.16f, 0.70f, 0.94f, 1f));
        classColors.put(2, new ColorRGBA(0.16f, 0.31f, 0.94f, 1f));
        classColors.put(3, new ColorRGBA(0.86f, 0.16f, 0.86f, 1f));
    }

    public void configure(int screenWidth, int screenHeight, int sourceWidth, int sourceHeight) {
        this.screenWidth = screenWidth;
        this.screenHeight = screenHeight;
        this.sourceWidth = sourceWidth;
        this.sourceHeight = sourceHeight;
    }

    public void update(List<Detection> detections) {
        overlayNode.detachAllChildren();
        if (detections == null || detections.isEmpty()) return;
        if (sourceWidth <= 0 || sourceHeight <= 0) return;
        float sx = screenWidth / (float) sourceWidth;
        float sy = screenHeight / (float) sourceHeight;
        for (Detection d : detections) {
            BoundingBox b = d.getBox();
            float x = b.getX() * sx;
            float w = b.getWidth() * sx;
            float h = b.getHeight() * sy;
            float yTop = screenHeight - b.getY() * sy;
            float y = yTop - h;
            ColorRGBA c = classColors.getOrDefault(d.getClassId(), ColorRGBA.Green);
            attachBorder(x, y, w, h, c, 2f);
            attachLabel(d.getLabel() + " " + Math.round(d.getConfidence() * 100f) + "%",
                    x, yTop, c);
        }
    }

    private void attachBorder(float x, float y, float w, float h, ColorRGBA color, float thickness) {
        attachRect(x, y, w, thickness, color);
        attachRect(x, y + h - thickness, w, thickness, color);
        attachRect(x, y, thickness, h, color);
        attachRect(x + w - thickness, y, thickness, h, color);
    }

    private void attachRect(float x, float y, float w, float h, ColorRGBA color) {
        Geometry g = new Geometry("rect", new Quad(w, h));
        Material m = new Material(assetManager, "Common/MatDefs/Misc/Unshaded.j3md");
        m.setColor("Color", color);
        m.getAdditionalRenderState().setBlendMode(RenderState.BlendMode.Alpha);
        g.setMaterial(m);
        g.setLocalTranslation(new Vector3f(x, y, 0f));
        overlayNode.attachChild(g);
    }

    private void attachLabel(String text, float x, float yTop, ColorRGBA color) {
        BitmapText t = new BitmapText(font);
        t.setSize(18f);
        t.setColor(color);
        t.setText(text);
        t.setLocalTranslation(x + 4f, yTop + 2f + t.getLineHeight(), 0f);
        overlayNode.attachChild(t);
    }
}
