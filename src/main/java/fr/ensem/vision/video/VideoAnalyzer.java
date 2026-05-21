package fr.ensem.vision.video;

import fr.ensem.vision.config.ClassMapping;
import fr.ensem.vision.detection.Detection;
import fr.ensem.vision.detection.DetectionFilter;
import fr.ensem.vision.inference.InferenceEngine;
import fr.ensem.vision.tracker.DetectionAggregator;
import fr.ensem.vision.tracker.SequentialSignTracker;
import fr.ensem.vision.tracker.TemporalFilter;
import fr.ensem.vision.util.AtomicJsonWriter;
import fr.ensem.vision.util.CheckpointManager;
import fr.ensem.vision.util.EtaCalculator;
import fr.ensem.vision.util.ProgressLogger;
import fr.ensem.vision.util.ResourceMonitor;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class VideoAnalyzer {

    private static final Logger LOG = LoggerFactory.getLogger(VideoAnalyzer.class);

    public static final class Config {
        public Path input;
        public Path output;
        public Path reportJson;
        public List<String> expectedSequence;
        public int maxFrames = -1;
        public int frameSkip = 1;
        public int windowSize = 15;
        public int minDetections = 5;
        public float minMeanConfidence = 0.5f;
        public float confidenceThreshold = 0.3f;
        public DetectionFilter classFilter;
    }

    public static final class Summary {
        public String engine;
        public String input;
        public String output;
        public List<String> expectedSequence;
        public List<SequentialSignTracker.Transition> transitions;
        public long framesProcessed;
        public long framesAccepted;
        public double averageFps;
        public double totalSeconds;
        public Map<String, Long> detectionCountsByClass;
    }

    private final InferenceEngine engine;
    private final ClassMapping mapping;
    private final FrameAnnotator annotator;
    private final CheckpointManager checkpoint;
    private final ResourceMonitor resources;

    public VideoAnalyzer(InferenceEngine engine, ClassMapping mapping,
                         FrameAnnotator annotator,
                         CheckpointManager checkpoint,
                         ResourceMonitor resources) {
        this.engine = engine;
        this.mapping = mapping;
        this.annotator = annotator;
        this.checkpoint = checkpoint;
        this.resources = resources;
    }

    public Summary process(Config cfg) {
        Summary summary = new Summary();
        summary.engine = engine.name();
        summary.input = cfg.input.toString();
        summary.output = cfg.output == null ? "" : cfg.output.toString();
        summary.expectedSequence = cfg.expectedSequence;
        summary.detectionCountsByClass = new LinkedHashMap<>();
        for (int i = 0; i < mapping.size(); i++) {
            summary.detectionCountsByClass.put(mapping.byId(i).getShortName(), 0L);
        }

        boolean useTracker = cfg.expectedSequence != null && !cfg.expectedSequence.isEmpty();
        DetectionAggregator agg = useTracker ? new DetectionAggregator(cfg.windowSize) : null;
        TemporalFilter tf = useTracker ? new TemporalFilter(cfg.minDetections, cfg.minMeanConfidence) : null;
        SequentialSignTracker tracker = useTracker
                ? new SequentialSignTracker(cfg.expectedSequence, mapping, agg, tf) : null;

        FrameSampler sampler = new FrameSampler(cfg.frameSkip);
        EtaCalculator eta = new EtaCalculator(20);

        try (VideoReader reader = new VideoReader(cfg.input)) {
            long total = reader.totalFrames();
            if (cfg.maxFrames > 0) total = Math.min(total, cfg.maxFrames);
            eta.setTotal(total);

            try (VideoWriter writer = cfg.output != null ? new VideoWriter(cfg.output, reader.width(), reader.height(), reader.fps()) : null;
                 ProgressLogger pl = new ProgressLogger("video[" + engine.name() + "]", eta, resources, 5)) {
                pl.start();
                if (checkpoint != null) checkpoint.loadOrCreate("video-" + cfg.input.getFileName(), total);

                long startNs = System.nanoTime();
                long framesProcessed = 0;
                long framesAccepted = 0;

                while (true) {
                    if (cfg.maxFrames > 0 && framesProcessed >= cfg.maxFrames) break;
                    Mat frame;
                    try (PointerScope scope = new PointerScope()) {
                        frame = reader.readFrame();
                        if (frame == null) break;
                        framesProcessed++;
                        long fIdx = reader.currentFrameIndex();
                        double tsMs = reader.currentTimestampMs();

                        boolean accept = sampler.accept();
                        List<Detection> detections = new ArrayList<>();
                        if (accept) {
                            framesAccepted++;
                            List<Detection> raw = engine.infer(frame);
                            if (cfg.classFilter != null) raw = cfg.classFilter.apply(raw);
                            for (int i = 0; i < raw.size(); i++) {
                                Detection d = raw.get(i).withFrame(fIdx, tsMs);
                                detections.add(d);
                                String label = mapping.labelForId(d.getClassId());
                                summary.detectionCountsByClass.merge(label, 1L, Long::sum);
                            }
                            if (tracker != null) tracker.update(fIdx, tsMs, detections);
                        }

                        if (writer != null) {
                            annotator.drawDetections(frame, detections);
                            if (tracker != null) annotator.drawTrackerState(frame, tracker.currentStateLabel());
                            writer.write(frame);
                        }

                        frame.release();
                        frame.close();
                    }

                    eta.tick();
                    if (checkpoint != null) {
                        try { checkpoint.update(1L); } catch (IOException ignored) {}
                    }
                    if (resources != null) resources.maybeFreeMemory();
                }

                long elapsedNs = System.nanoTime() - startNs;
                summary.framesProcessed = framesProcessed;
                summary.framesAccepted = framesAccepted;
                summary.totalSeconds = elapsedNs / 1e9d;
                summary.averageFps = summary.totalSeconds > 0d ? framesProcessed / summary.totalSeconds : 0d;
            }
        } catch (Exception e) {
            LOG.error("Video processing failed for {}", cfg.input, e);
            throw new RuntimeException(e);
        }

        summary.transitions = tracker != null
                ? tracker.transitions()
                : java.util.Collections.emptyList();

        if (cfg.reportJson != null) {
            try {
                AtomicJsonWriter.write(cfg.reportJson, summary);
            } catch (IOException e) {
                LOG.warn("Cannot write summary: {}", e.getMessage());
            }
        }
        return summary;
    }
}
