package fr.ensem.vision.data;

import fr.ensem.vision.config.ClassMapping;
import fr.ensem.vision.detection.Detection;
import fr.ensem.vision.inference.InferenceEngine;
import fr.ensem.vision.util.CheckpointManager;
import fr.ensem.vision.util.EtaCalculator;
import fr.ensem.vision.util.ProgressLogger;
import fr.ensem.vision.util.ResourceMonitor;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

public final class AutoAnnotator {

    private static final Logger LOG = LoggerFactory.getLogger(AutoAnnotator.class);

    public static final class Config {
        public Path imagesDir;
        public Path labelsDir;
        public Path csvFile;
        public int maxImages = -1;
        public float minConfidence = 0.5f;
        public boolean keepOnlyMatchingCsvClass = true;
    }

    private final InferenceEngine engine;
    private final ClassMapping mapping;
    private final ClassificationToDetectionMapper csvMapper;
    private final CheckpointManager checkpoint;
    private final ResourceMonitor resources;

    public AutoAnnotator(InferenceEngine engine,
                         ClassMapping mapping,
                         CheckpointManager checkpoint,
                         ResourceMonitor resources) {
        this.engine = engine;
        this.mapping = mapping;
        this.csvMapper = new ClassificationToDetectionMapper(mapping);
        this.checkpoint = checkpoint;
        this.resources = resources;
    }

    public int run(Config cfg) throws IOException {
        Map<String, Set<Integer>> csvAllowed = csvMapper.mapPositives(new RoboflowCsvReader(cfg.csvFile).indexByFilename());
        List<Path> imgs = new ArrayList<>();
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(cfg.imagesDir, "*.{jpg,jpeg,png}")) {
            for (Path p : ds) imgs.add(p);
        }
        imgs.sort((a, b) -> a.getFileName().toString().compareTo(b.getFileName().toString()));
        if (cfg.maxImages > 0 && imgs.size() > cfg.maxImages) imgs = imgs.subList(0, cfg.maxImages);

        long total = imgs.size();
        if (checkpoint != null) checkpoint.loadOrCreate("auto-annotate-" + cfg.imagesDir.getFileName(), total);

        EtaCalculator eta = new EtaCalculator(20);
        eta.setTotal(total);
        ProgressLogger pl = new ProgressLogger("annot", eta, resources, 5);
        pl.start();

        int written = 0;
        int kept = 0;
        long startIndex = (checkpoint != null && checkpoint.state() != null) ? checkpoint.state().processed : 0L;

        for (long i = startIndex; i < total; i++) {
            Path img = imgs.get((int) i);
            String filename = img.getFileName().toString();
            Set<Integer> allowed = csvAllowed.get(filename);
            if (cfg.keepOnlyMatchingCsvClass && (allowed == null || allowed.isEmpty())) {
                eta.tick();
                if (checkpoint != null) checkpoint.update(1L);
                continue;
            }

            try (PointerScope scope = new PointerScope()) {
                Mat frame = imread(img.toAbsolutePath().toString());
                if (frame == null || frame.empty()) {
                    LOG.warn("Cannot read image: {}", img);
                    eta.tick();
                    if (checkpoint != null) checkpoint.update(1L);
                    continue;
                }
                int w = frame.cols();
                int h = frame.rows();
                List<Detection> raw = engine.infer(frame);
                List<Detection> filtered = new ArrayList<>();
                for (Detection d : raw) {
                    if (d.getConfidence() < cfg.minConfidence) continue;
                    if (cfg.keepOnlyMatchingCsvClass && !allowed.contains(d.getClassId())) continue;
                    filtered.add(d);
                }
                Path labelOut = cfg.labelsDir.resolve(stripExt(filename) + ".txt");
                YoloLabelWriter.write(labelOut, filtered, w, h);
                if (!filtered.isEmpty()) {
                    kept += filtered.size();
                    written++;
                }
                frame.release();
                frame.close();
            }

            eta.tick();
            if (checkpoint != null) checkpoint.update(1L);
            if (resources != null) resources.maybeFreeMemory();
        }

        pl.close();
        if (checkpoint != null) checkpoint.save();
        LOG.info("Auto-annotation done: imagesProcessed={} fileWritten={} bboxKept={}", total, written, kept);
        return written;
    }

    private static String stripExt(String n) {
        int dot = n.lastIndexOf('.');
        return dot < 0 ? n : n.substring(0, dot);
    }
}
