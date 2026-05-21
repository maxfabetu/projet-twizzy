package fr.ensem.vision.tracker;

import fr.ensem.vision.config.ClassMapping;
import fr.ensem.vision.detection.Detection;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

public final class SequentialSignTracker {

    private static final Logger LOG = LoggerFactory.getLogger(SequentialSignTracker.class);

    public static final class Transition {
        public final int stepIndex;
        public final String expectedClass;
        public final long frameIndex;
        public final double timestampMs;
        public final float meanConfidence;
        public final int supportingDetections;

        public Transition(int stepIndex, String expectedClass, long frameIndex,
                          double timestampMs, float meanConfidence, int supportingDetections) {
            this.stepIndex = stepIndex;
            this.expectedClass = expectedClass;
            this.frameIndex = frameIndex;
            this.timestampMs = timestampMs;
            this.meanConfidence = meanConfidence;
            this.supportingDetections = supportingDetections;
        }
    }

    private final List<String> sequence;
    private final ClassMapping mapping;
    private final DetectionAggregator aggregator;
    private final TemporalFilter filter;
    private final List<Transition> transitions = new ArrayList<>();
    private int currentStep = 0;

    public SequentialSignTracker(List<String> shortClassSequence,
                                 ClassMapping mapping,
                                 DetectionAggregator aggregator,
                                 TemporalFilter filter) {
        this.sequence = new ArrayList<>(shortClassSequence);
        this.mapping = mapping;
        this.aggregator = aggregator;
        this.filter = filter;
    }

    public void update(long frameIndex, double timestampMs, List<Detection> frameDetections) {
        aggregator.add(frameIndex, frameDetections);
        if (isDone()) return;
        String expected = sequence.get(currentStep);
        ClassMapping.Entry exp = mapping.byShort(expected);
        if (exp == null) return;
        List<Detection> classDets = aggregator.classDetections(exp.getId());
        if (filter.isConfirmed(classDets)) {
            float sum = 0f;
            for (Detection d : classDets) sum += d.getConfidence();
            float mean = sum / classDets.size();
            Transition t = new Transition(currentStep, expected, frameIndex, timestampMs, mean, classDets.size());
            transitions.add(t);
            LOG.info("Tracker transition: step={} class={} frame={} t={}ms conf={} support={}",
                    currentStep, expected, frameIndex, String.format("%.0f", timestampMs),
                    String.format("%.2f", mean), classDets.size());
            currentStep++;
            aggregator.clear();
        }
    }

    public boolean isDone() {
        return currentStep >= sequence.size();
    }

    public int currentStep() { return currentStep; }
    public List<String> sequence() { return sequence; }
    public List<Transition> transitions() { return transitions; }

    public String currentStateLabel() {
        if (isDone()) return "DONE";
        return "WAIT_" + sequence.get(currentStep);
    }
}
