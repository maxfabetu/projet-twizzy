package fr.ensem.vision.tracker;

import fr.ensem.vision.config.ClassMapping;
import fr.ensem.vision.detection.BoundingBox;
import fr.ensem.vision.detection.Detection;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SequentialSignTrackerTest {

    private static Detection mk(int classId, float conf) {
        return new Detection(new BoundingBox(10, 10, 50, 50), classId, String.valueOf(classId), conf);
    }

    @Test
    void completesSequence907050() {
        ClassMapping mapping = ClassMapping.defaultMapping();
        SequentialSignTracker tracker = new SequentialSignTracker(
                Arrays.asList("90", "70", "50"),
                mapping,
                new DetectionAggregator(15),
                new TemporalFilter(5, 0.5f));

        feed(tracker, 1L, 100L, 2, 0.8f);
        assertEquals(1, tracker.currentStep());
        feed(tracker, 101L, 200L, 1, 0.7f);
        assertEquals(2, tracker.currentStep());
        feed(tracker, 201L, 300L, 0, 0.75f);
        assertTrue(tracker.isDone());
        assertEquals(3, tracker.transitions().size());
        assertEquals("90", tracker.transitions().get(0).expectedClass);
        assertEquals("70", tracker.transitions().get(1).expectedClass);
        assertEquals("50", tracker.transitions().get(2).expectedClass);
    }

    @Test
    void doesNotTransitionWithoutEnoughDetections() {
        ClassMapping mapping = ClassMapping.defaultMapping();
        SequentialSignTracker tracker = new SequentialSignTracker(
                Collections.singletonList("110"),
                mapping,
                new DetectionAggregator(15),
                new TemporalFilter(5, 0.6f));

        for (int i = 0; i < 3; i++) {
            tracker.update(i, i * 10d, Arrays.asList(mk(3, 0.9f)));
        }
        assertFalse(tracker.isDone());
    }

    @Test
    void noisyOtherClassesDoNotAdvance() {
        ClassMapping mapping = ClassMapping.defaultMapping();
        SequentialSignTracker tracker = new SequentialSignTracker(
                Collections.singletonList("110"),
                mapping,
                new DetectionAggregator(15),
                new TemporalFilter(5, 0.6f));

        for (int i = 0; i < 10; i++) {
            tracker.update(i, i * 10d, Arrays.asList(mk(0, 0.9f), mk(2, 0.9f)));
        }
        assertFalse(tracker.isDone());
    }

    private static void feed(SequentialSignTracker tracker, long startFrame, double tsStart,
                             int classId, float conf) {
        for (int i = 0; i < 6; i++) {
            List<Detection> ds = new ArrayList<>();
            ds.add(mk(classId, conf));
            tracker.update(startFrame + i, tsStart + i * 33d, ds);
        }
    }
}
