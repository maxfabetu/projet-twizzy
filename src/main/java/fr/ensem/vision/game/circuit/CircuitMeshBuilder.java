package fr.ensem.vision.game.circuit;

import com.jme3.asset.AssetManager;
import com.jme3.material.Material;
import com.jme3.math.ColorRGBA;
import com.jme3.math.Vector2f;
import com.jme3.math.Vector3f;
import com.jme3.scene.Geometry;
import com.jme3.scene.Mesh;
import com.jme3.scene.Node;
import com.jme3.scene.VertexBuffer;
import com.jme3.scene.shape.Quad;
import com.jme3.util.BufferUtils;

import java.util.List;

public final class CircuitMeshBuilder {

    private final AssetManager assetManager;

    public CircuitMeshBuilder(AssetManager assetManager) {
        this.assetManager = assetManager;
    }

    public Node build(Circuit circuit) {
        Node root = new Node("circuit");
        root.attachChild(buildGround());
        root.attachChild(buildRoad(circuit));
        root.attachChild(buildLineStrip(circuit, true));
        root.attachChild(buildLineStrip(circuit, false));
        return root;
    }

    private Geometry buildGround() {
        float size = 2000f;
        Quad q = new Quad(size, size);
        Geometry g = new Geometry("ground", q);
        Material m = new Material(assetManager, "Common/MatDefs/Misc/Unshaded.j3md");
        m.setColor("Color", new ColorRGBA(0.18f, 0.45f, 0.18f, 1f));
        g.setMaterial(m);
        g.rotate(-(float) Math.PI / 2f, 0f, 0f);
        g.setLocalTranslation(-size / 2f, -0.02f, size / 2f);
        return g;
    }

    private Geometry buildRoad(Circuit circuit) {
        List<Vector3f> left = circuit.getLeftEdge();
        List<Vector3f> right = circuit.getRightEdge();
        int n = left.size();
        int quadCount = n - 1;
        float[] positions = new float[n * 2 * 3];
        float[] uvs = new float[n * 2 * 2];
        int[] indices = new int[quadCount * 6];
        for (int i = 0; i < n; i++) {
            int b = i * 6;
            positions[b] = left.get(i).x;
            positions[b + 1] = left.get(i).y;
            positions[b + 2] = left.get(i).z;
            positions[b + 3] = right.get(i).x;
            positions[b + 4] = right.get(i).y;
            positions[b + 5] = right.get(i).z;
            uvs[i * 4] = 0f;
            uvs[i * 4 + 1] = i;
            uvs[i * 4 + 2] = 1f;
            uvs[i * 4 + 3] = i;
        }
        for (int i = 0; i < quadCount; i++) {
            int v0 = i * 2;
            int v1 = i * 2 + 1;
            int v2 = i * 2 + 2;
            int v3 = i * 2 + 3;
            int b = i * 6;
            indices[b] = v0; indices[b + 1] = v2; indices[b + 2] = v1;
            indices[b + 3] = v1; indices[b + 4] = v2; indices[b + 5] = v3;
        }
        Mesh mesh = new Mesh();
        mesh.setBuffer(VertexBuffer.Type.Position, 3, BufferUtils.createFloatBuffer(positions));
        mesh.setBuffer(VertexBuffer.Type.TexCoord, 2, BufferUtils.createFloatBuffer(uvs));
        mesh.setBuffer(VertexBuffer.Type.Index, 3, BufferUtils.createIntBuffer(indices));
        mesh.updateBound();
        Geometry geom = new Geometry("road", mesh);
        Material mat = new Material(assetManager, "Common/MatDefs/Misc/Unshaded.j3md");
        mat.setColor("Color", new ColorRGBA(0.18f, 0.18f, 0.20f, 1f));
        geom.setMaterial(mat);
        geom.setLocalTranslation(0f, 0.01f, 0f);
        return geom;
    }

    private Geometry buildLineStrip(Circuit circuit, boolean leftSide) {
        List<Vector3f> edge = leftSide ? circuit.getLeftEdge() : circuit.getRightEdge();
        List<Vector3f> center = circuit.getCenterline();
        int n = edge.size();
        float inset = 0.4f;
        float lineWidth = 0.15f;
        float[] positions = new float[n * 2 * 3];
        int[] indices = new int[(n - 1) * 6];
        for (int i = 0; i < n; i++) {
            Vector3f e = edge.get(i);
            Vector3f c = center.get(i);
            Vector3f toCenter = c.subtract(e).normalizeLocal();
            Vector3f inner = e.add(toCenter.mult(inset));
            Vector3f outer = inner.add(toCenter.mult(lineWidth));
            int b = i * 6;
            positions[b] = inner.x; positions[b + 1] = 0.02f; positions[b + 2] = inner.z;
            positions[b + 3] = outer.x; positions[b + 4] = 0.02f; positions[b + 5] = outer.z;
        }
        for (int i = 0; i < n - 1; i++) {
            int v0 = i * 2;
            int v1 = i * 2 + 1;
            int v2 = i * 2 + 2;
            int v3 = i * 2 + 3;
            int b = i * 6;
            indices[b] = v0; indices[b + 1] = v2; indices[b + 2] = v1;
            indices[b + 3] = v1; indices[b + 4] = v2; indices[b + 5] = v3;
        }
        Mesh mesh = new Mesh();
        mesh.setBuffer(VertexBuffer.Type.Position, 3, BufferUtils.createFloatBuffer(positions));
        mesh.setBuffer(VertexBuffer.Type.Index, 3, BufferUtils.createIntBuffer(indices));
        mesh.updateBound();
        Geometry geom = new Geometry(leftSide ? "lineLeft" : "lineRight", mesh);
        Material mat = new Material(assetManager, "Common/MatDefs/Misc/Unshaded.j3md");
        mat.setColor("Color", ColorRGBA.White);
        geom.setMaterial(mat);
        return geom;
    }

    public static Vector2f flat(Vector3f v) {
        return new Vector2f(v.x, v.z);
    }
}
