package fr.ensem.vision.game.sign;

import com.jme3.asset.AssetManager;
import com.jme3.asset.AssetNotFoundException;
import com.jme3.asset.TextureKey;
import com.jme3.texture.Texture;

import java.util.HashMap;
import java.util.Map;

public final class SignTextureLoader {

    private final AssetManager assetManager;
    private final Map<String, Texture> cache = new HashMap<>();

    public SignTextureLoader(AssetManager assetManager) {
        this.assetManager = assetManager;
    }

    public Texture get(String shortName) {
        Texture t = cache.get(shortName);
        if (t != null) return t;
        TextureKey key = new TextureKey("signs/" + shortName + ".png", false);
        key.setGenerateMips(true);
        try {
            t = assetManager.loadTexture(key);
            t.setMagFilter(Texture.MagFilter.Bilinear);
            t.setMinFilter(Texture.MinFilter.Trilinear);
        } catch (AssetNotFoundException e) {
            throw new RuntimeException("Sign texture not found: " + key.getName(), e);
        }
        cache.put(shortName, t);
        return t;
    }
}
