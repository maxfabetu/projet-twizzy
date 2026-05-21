package fr.ensem.vision.eval;

import fr.ensem.vision.config.ClassMapping;
import fr.ensem.vision.detection.DetectionFilter;
import fr.ensem.vision.inference.InferenceEngine;
import fr.ensem.vision.tracker.SequentialSignTracker;
import fr.ensem.vision.video.FrameAnnotator;
import fr.ensem.vision.video.VideoAnalyzer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class MethodComparator {

    private static final Logger LOG = LoggerFactory.getLogger(MethodComparator.class);

    public static final class VideoOutcome {
        public String videoFile;
        public List<String> expectedSequence;
        public List<String> detectedSequence;
        public boolean sequenceMatch;
        public double averageFps;
        public List<SequentialSignTracker.Transition> transitions;
    }

    public static final class EngineComparison {
        public String engineName;
        public Evaluator.Report evalReport;
        public FpsBenchmark.Result fpsBenchmark;
        public List<VideoOutcome> videoOutcomes = new ArrayList<>();
    }

    public static final class Report {
        public Map<String, EngineComparison> byEngine = new LinkedHashMap<>();
        public String winnerByMap;
        public String winnerByFps;
    }

    public Report compare(List<InferenceEngine> engines,
                          ClassMapping mapping,
                          Path testImages,
                          Path testLabels,
                          int testMaxSamples,
                          List<VideoSpec> videoSpecs) throws IOException {

        Report report = new Report();
        double bestMap = -1d;
        double bestFps = -1d;

        for (InferenceEngine e : engines) {
            EngineComparison ec = new EngineComparison();
            ec.engineName = e.name();

            Evaluator.Report eval = new Evaluator(e, mapping).run(testImages, testLabels, testMaxSamples);
            ec.evalReport = eval;
            ec.fpsBenchmark = FpsBenchmark.runOnDirectory(e, testImages, Math.min(200, testMaxSamples > 0 ? testMaxSamples : 200));

            for (VideoSpec spec : videoSpecs) {
                VideoAnalyzer analyzer = new VideoAnalyzer(e, mapping, new FrameAnnotator(), null, null);
                VideoAnalyzer.Config cfg = new VideoAnalyzer.Config();
                cfg.input = spec.input;
                cfg.output = spec.outputOverride;
                cfg.reportJson = spec.reportJson;
                cfg.expectedSequence = spec.expectedSequence;
                cfg.maxFrames = spec.maxFrames;
                cfg.frameSkip = spec.frameSkip;
                cfg.windowSize = spec.windowSize;
                cfg.minDetections = spec.minDetections;
                cfg.minMeanConfidence = spec.minMeanConfidence;
                cfg.confidenceThreshold = spec.confidenceThreshold;
                cfg.classFilter = spec.classFilter;
                VideoAnalyzer.Summary s = analyzer.process(cfg);
                VideoOutcome vo = new VideoOutcome();
                vo.videoFile = spec.input.getFileName().toString();
                vo.expectedSequence = spec.expectedSequence;
                vo.detectedSequence = new ArrayList<>();
                for (SequentialSignTracker.Transition t : s.transitions) vo.detectedSequence.add(t.expectedClass);
                vo.sequenceMatch = vo.detectedSequence.equals(vo.expectedSequence);
                vo.averageFps = s.averageFps;
                vo.transitions = s.transitions;
                ec.videoOutcomes.add(vo);
            }

            report.byEngine.put(e.name(), ec);
            if (eval.mapAt50 > bestMap) {
                bestMap = eval.mapAt50;
                report.winnerByMap = e.name();
            }
            if (ec.fpsBenchmark != null && ec.fpsBenchmark.fps > bestFps) {
                bestFps = ec.fpsBenchmark.fps;
                report.winnerByFps = e.name();
            }
        }
        LOG.info("Comparison complete: winnerByMap={} winnerByFps={}", report.winnerByMap, report.winnerByFps);
        return report;
    }

    public static final class VideoSpec {
        public Path input;
        public Path outputOverride;
        public Path reportJson;
        public List<String> expectedSequence;
        public int maxFrames = -1;
        public int frameSkip = 1;
        public int windowSize = 15;
        public int minDetections = 5;
        public float minMeanConfidence = 0.5f;
        public float confidenceThreshold = 0.3f;
        public DetectionFilter classFilter;

        public static VideoSpec of(Path input, Path output, Path report, List<String> seq) {
            VideoSpec s = new VideoSpec();
            s.input = input;
            s.outputOverride = output;
            s.reportJson = report;
            s.expectedSequence = seq;
            return s;
        }

        public VideoSpec withFilter(ClassMapping mapping) {
            Set<Integer> ids = new HashSet<>();
            for (ClassMapping.Entry e : mapping.getEntries()) ids.add(e.getId());
            this.classFilter = new DetectionFilter(ids, this.confidenceThreshold);
            return this;
        }
    }
}
