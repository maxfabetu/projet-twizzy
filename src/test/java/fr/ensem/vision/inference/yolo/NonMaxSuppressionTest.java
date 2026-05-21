package fr.ensem.vision.inference.yolo;

import fr.ensem.vision.detection.BoundingBox;
import fr.ensem.vision.detection.Detection;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class NonMaxSuppressionTest {

    @Test
    void suppressesHeavilyOverlappingSameClass() {
        NonMaxSuppression nms = new NonMaxSuppression(0.4f);
        Detection a = new Detection(new BoundingBox(0, 0, 100, 100), 0, "50", 0.9f);
        Detection b = new Detection(new BoundingBox(5, 5, 100, 100), 0, "50", 0.8f);
        Detection c = new Detection(new BoundingBox(200, 200, 50, 50), 0, "50", 0.7f);

        List<Detection> kept = nms.apply(Arrays.asList(a, b, c));
        assertEquals(2, kept.size());
    }

    @Test
    void keepsDifferentClassesEvenIfOverlapping() {
        NonMaxSuppression nms = new NonMaxSuppression(0.4f);
        Detection a = new Detection(new BoundingBox(0, 0, 100, 100), 0, "50", 0.9f);
        Detection b = new Detection(new BoundingBox(5, 5, 100, 100), 1, "70", 0.8f);

        List<Detection> kept = nms.apply(Arrays.asList(a, b));
        assertEquals(2, kept.size());
    }
}
