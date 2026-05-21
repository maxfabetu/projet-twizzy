package fr.ensem.vision.game.sign;

import com.jme3.asset.AssetManager;
import com.jme3.math.Vector3f;
import com.jme3.scene.Node;
import fr.ensem.vision.game.GameConfig;
import fr.ensem.vision.game.circuit.Circuit;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public final class SignPlacer {

    private static final String[] CLASSES = { "50", "70", "90", "110" };

    private final AssetManager assetManager;
    private final SignTextureLoader textureLoader;
    private final GameConfig cfg;

    public SignPlacer(AssetManager assetManager, SignTextureLoader textureLoader, GameConfig cfg) {
        this.assetManager = assetManager;
        this.textureLoader = textureLoader;
        this.cfg = cfg;
    }

    public List<SignEntity> place(Circuit circuit, long seed, Node parent) {
        List<SignEntity> placed = new ArrayList<>();
        Random rng = new Random(seed ^ 0x53C3L);
        float total = circuit.getTotalLength();
        if (total < cfg.signMinDistance) return placed;
        float cursor = cfg.signMinDistance;
        Vector3f pos = new Vector3f();
        while (cursor < total - cfg.signMinDistance) {
            float t = cursor;
            circuit.sampleAt(t, pos);
            float heading = circuit.headingAt(t);
            float nx = (float) Math.cos(heading);
            float nz = -(float) Math.sin(heading);
            Vector3f signPos = new Vector3f(pos.x + nx * cfg.signLateralOffset, 0f, pos.z + nz * cfg.signLateralOffset);
            float yaw = heading + (float) Math.PI;
            String cls = CLASSES[rng.nextInt(CLASSES.length)];
            SignEntity s = new SignEntity(assetManager, textureLoader, cls, signPos, yaw, cfg.signSize, cfg.signHeight);
            parent.attachChild(s.getNode());
            placed.add(s);
            cursor += cfg.signMinDistance + rng.nextFloat() * (cfg.signMaxDistance - cfg.signMinDistance);
        }
        return placed;
    }
}
