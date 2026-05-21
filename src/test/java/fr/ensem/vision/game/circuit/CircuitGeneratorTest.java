package fr.ensem.vision.game.circuit;

import com.jme3.math.Vector3f;
import fr.ensem.vision.game.GameConfig;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CircuitGeneratorTest {

    @Test
    void seedIsReproducible() {
        GameConfig cfg = new GameConfig();
        cfg.segmentCount = 12;
        CircuitGenerator g = new CircuitGenerator(cfg);
        Circuit a = g.generate(42L);
        Circuit b = g.generate(42L);
        assertEquals(a.getSegments().size(), b.getSegments().size());
        for (int i = 0; i < a.getSegments().size(); i++) {
            assertEquals(a.getSegments().get(i).type, b.getSegments().get(i).type);
        }
        assertEquals(a.sampleCount(), b.sampleCount());
    }

    @Test
    void respectsSegmentCount() {
        GameConfig cfg = new GameConfig();
        cfg.segmentCount = 20;
        Circuit c = new CircuitGenerator(cfg).generate(1L);
        assertEquals(20, c.getSegments().size());
    }

    @Test
    void buildsEdgesAndCenterlineSameSize() {
        GameConfig cfg = new GameConfig();
        cfg.segmentCount = 8;
        Circuit c = new CircuitGenerator(cfg).generate(7L);
        assertEquals(c.getCenterline().size(), c.getLeftEdge().size());
        assertEquals(c.getCenterline().size(), c.getRightEdge().size());
        assertEquals(c.getCenterline().size(), c.getHeadings().size());
        assertTrue(c.getTotalLength() > 0);
    }

    @Test
    void sampleAtStartReturnsOrigin() {
        GameConfig cfg = new GameConfig();
        cfg.segmentCount = 6;
        Circuit c = new CircuitGenerator(cfg).generate(123L);
        Vector3f p = c.sampleAt(0f, null);
        assertNotNull(p);
        assertEquals(0f, p.length(), 1e-3f);
    }

    @Test
    void roadHasExpectedWidth() {
        GameConfig cfg = new GameConfig();
        cfg.segmentCount = 5;
        cfg.roadWidth = 8f;
        Circuit c = new CircuitGenerator(cfg).generate(2L);
        List<Vector3f> l = c.getLeftEdge();
        List<Vector3f> r = c.getRightEdge();
        for (int i = 0; i < l.size(); i++) {
            float dist = l.get(i).distance(r.get(i));
            assertTrue(Math.abs(dist - 8f) < 1e-2f);
        }
    }
}
