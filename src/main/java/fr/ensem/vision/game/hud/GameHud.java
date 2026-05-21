package fr.ensem.vision.game.hud;

import com.jme3.font.BitmapFont;
import com.jme3.font.BitmapText;
import com.jme3.math.ColorRGBA;
import com.jme3.scene.Node;

public final class GameHud {

    private final Node parent;
    private final BitmapFont font;
    private final BitmapText speedText;
    private final BitmapText limitText;
    private final BitmapText modeText;
    private final BitmapText perfText;
    private final BitmapText helpText;

    private int screenWidth;
    private int screenHeight;

    public GameHud(Node parent, BitmapFont font) {
        this.parent = parent;
        this.font = font;
        this.speedText = newText(48f, ColorRGBA.White);
        this.limitText = newText(22f, new ColorRGBA(1f, 0.85f, 0.2f, 1f));
        this.modeText = newText(22f, new ColorRGBA(0.6f, 0.95f, 0.6f, 1f));
        this.perfText = newText(16f, new ColorRGBA(0.8f, 0.8f, 0.85f, 1f));
        this.helpText = newText(14f, new ColorRGBA(0.7f, 0.7f, 0.78f, 1f));
        helpText.setText("Z/W/UP=accel  S/DOWN=brake  Q/A/LEFT=tourner  D/RIGHT=tourner  SPACE=brake hard  TAB=auto  R=regen  ESC=quitter");
        parent.attachChild(speedText);
        parent.attachChild(limitText);
        parent.attachChild(modeText);
        parent.attachChild(perfText);
        parent.attachChild(helpText);
    }

    private BitmapText newText(float size, ColorRGBA color) {
        BitmapText t = new BitmapText(font);
        t.setSize(size);
        t.setColor(color);
        return t;
    }

    public void layout(int screenWidth, int screenHeight) {
        this.screenWidth = screenWidth;
        this.screenHeight = screenHeight;
        float topMargin = 24f;
        float leftMargin = 24f;
        speedText.setLocalTranslation(leftMargin, screenHeight - topMargin, 0f);
        limitText.setLocalTranslation(leftMargin, screenHeight - topMargin - 56f, 0f);
        modeText.setLocalTranslation(leftMargin, screenHeight - topMargin - 84f, 0f);
        perfText.setLocalTranslation(leftMargin, screenHeight - topMargin - 110f, 0f);
        helpText.setLocalTranslation(leftMargin, 32f, 0f);
    }

    public void update(float speedKmh, float limitKmh, boolean autonomous,
                       double renderFps, double detectionFps,
                       String lastClass, float lastConf) {
        speedText.setText(String.format("%.0f km/h", Math.max(0f, speedKmh)));
        limitText.setText("Limite: " + (int) limitKmh + " km/h");
        modeText.setText("Mode: " + (autonomous ? "AUTONOME (TAB)" : "MANUEL (TAB)"));
        String last = lastClass != null ? (lastClass + " (" + Math.round(lastConf * 100f) + "%)") : "-";
        perfText.setText(String.format("Render %.0f fps | Detect %.0f fps | Dernier: %s",
                renderFps, detectionFps, last));
    }

    public int getScreenWidth() { return screenWidth; }
    public int getScreenHeight() { return screenHeight; }
}
