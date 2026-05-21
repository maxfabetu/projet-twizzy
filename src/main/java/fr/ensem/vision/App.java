package fr.ensem.vision;

import fr.ensem.vision.config.AppConfig;
import fr.ensem.vision.config.ClassMapping;
import fr.ensem.vision.data.AutoAnnotator;
import fr.ensem.vision.data.DataYamlWriter;
import fr.ensem.vision.data.DatasetValidator;
import fr.ensem.vision.data.YoloLabelWriter;
import fr.ensem.vision.detection.DetectionFilter;
import fr.ensem.vision.eval.Evaluator;
import fr.ensem.vision.eval.FpsBenchmark;
import fr.ensem.vision.eval.MethodComparator;
import fr.ensem.vision.eval.ReportWriter;
import fr.ensem.vision.inference.InferenceEngine;
import fr.ensem.vision.inference.haarcnn.HaarCnnInferenceEngine;
import fr.ensem.vision.inference.yolo.YoloOnnxInferenceEngine;
import fr.ensem.vision.util.AtomicJsonWriter;
import fr.ensem.vision.util.CheckpointManager;
import fr.ensem.vision.util.ResourceMonitor;
import fr.ensem.vision.game.GameApp;
import fr.ensem.vision.game.GameConfig;
import fr.ensem.vision.video.FrameAnnotator;
import fr.ensem.vision.video.VideoAnalyzer;
import fr.ensem.vision.video.WebcamAnalyzer;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imencode;

@Command(name = "vision-ensem", mixinStandardHelpOptions = true, version = "1.0.0",
        subcommands = {
                App.PrepareData.class,
                App.InferVideo.class,
                App.Evaluate.class,
                App.Compare.class,
                App.Bench.class,
                App.Smoke.class,
                App.Webcam.class,
                App.Game.class
        })
public final class App implements Callable<Integer> {

    private static final Logger LOG = LoggerFactory.getLogger(App.class);

    public static void main(String[] args) {
        int exit = new CommandLine(new App()).execute(args);
        System.exit(exit);
    }

    @Override
    public Integer call() {
        CommandLine.usage(this, System.out);
        return 0;
    }

    private static InferenceEngine buildEngine(String method, AppConfig cfg, ClassMapping mapping, Boolean cudaOverride) throws Exception {
        boolean cuda = cudaOverride != null ? cudaOverride : cfg.getBoolean("yolo.useCuda", true);
        switch (method.toLowerCase()) {
            case "yolo":
            case "yolov8":
                return new YoloOnnxInferenceEngine(
                        Paths.get(cfg.getString("yolo.modelPath", "models/yolov8n_signs.onnx")),
                        mapping,
                        cfg.getInt("yolo.inputSize", 640),
                        (float) cfg.getDouble("yolo.confidenceThreshold", 0.30),
                        (float) cfg.getDouble("yolo.nmsIouThreshold", 0.45),
                        cuda
                );
            case "haarcnn":
            case "haar":
                return new HaarCnnInferenceEngine(
                        Paths.get(cfg.getString("haarCnn.cascadePath", "models/haar_signs.xml")),
                        Paths.get(cfg.getString("haarCnn.cnnPath", "models/cnn_classifier.onnx")),
                        mapping,
                        cfg.getInt("haarCnn.cnnInputSize", 64),
                        cfg.getInt("haarCnn.minRoi", 24),
                        cfg.getInt("haarCnn.maxRoi", 200),
                        (float) cfg.getDouble("haarCnn.cnnConfThreshold", 0.55),
                        cuda
                );
            default:
                throw new IllegalArgumentException("Unknown method: " + method);
        }
    }

    private static DetectionFilter classFilter(ClassMapping mapping, float minConf) {
        Set<Integer> ids = new HashSet<>();
        for (ClassMapping.Entry e : mapping.getEntries()) ids.add(e.getId());
        return new DetectionFilter(ids, minConf);
    }

    @Command(name = "smoke", description = "Open the input video and save first frame as JPG")
    public static class Smoke implements Callable<Integer> {
        @Option(names = {"--input", "-i"}, required = true) Path input;
        @Option(names = {"--output", "-o"}) Path output;

        @Override
        public Integer call() throws Exception {
            Path out = output != null ? output : Paths.get("logs/" + input.getFileName() + ".frame0.jpg");
            if (out.getParent() != null) Files.createDirectories(out.getParent());
            try (fr.ensem.vision.video.VideoReader r = new fr.ensem.vision.video.VideoReader(input)) {
                Mat first = r.readFrame();
                if (first == null) {
                    LOG.error("No frame read from {}", input);
                    return 2;
                }
                BytePointer ext = new BytePointer(".jpg");
                BytePointer buf = new BytePointer();
                boolean ok = imencode(ext, first, buf);
                if (!ok) {
                    LOG.error("imencode failed (type={} channels={} depth={})",
                            first.type(), first.channels(), first.depth());
                    ext.deallocate(); buf.deallocate();
                    first.release(); first.close();
                    return 3;
                }
                byte[] bytes = new byte[(int) buf.limit()];
                buf.get(bytes);
                Files.write(out, bytes);
                LOG.info("Smoke ok: wrote {} ({}x{} {} bytes)", out.toAbsolutePath(),
                        first.cols(), first.rows(), bytes.length);
                ext.deallocate(); buf.deallocate();
                first.release(); first.close();
            }
            return 0;
        }
    }

    @Command(name = "prepare-data", description = "Generate YOLO labels from Roboflow CSV by pseudo-labelling")
    public static class PrepareData implements Callable<Integer> {
        @Option(names = "--split", description = "train | valid | test | all", defaultValue = "all") String split = "all";
        @Option(names = "--max-images", description = "Limit images per split (smoke)") int maxImages = -1;
        @Option(names = "--method", description = "Engine used for pseudo-labelling", defaultValue = "yolo") String method = "yolo";
        @Option(names = "--no-cuda") boolean noCuda;
        @Option(names = "--min-conf", description = "Min confidence to keep bbox", defaultValue = "0.5") float minConf;

        @Override
        public Integer call() throws Exception {
            AppConfig cfg = AppConfig.load();
            ClassMapping mapping = cfg.classMapping();
            Path raw = Paths.get(cfg.getString("paths.rawDataset", "data/raw/BDD_Roboflow_Tazi/BDD_Roboflow_Tazi"));
            Path yoloRoot = Paths.get(cfg.getString("paths.yoloDataset", "data/yolo"));
            YoloLabelWriter.writeClassesTxt(yoloRoot.resolve("classes.txt"), mapping.shortNamesInOrder());
            DataYamlWriter.write(yoloRoot.resolve("data.yaml"), yoloRoot, mapping);

            List<String> splits = "all".equalsIgnoreCase(split)
                    ? Arrays.asList("train", "valid", "test") : List.of(split);

            try (InferenceEngine engine = buildEngine(method, cfg, mapping, !noCuda)) {
                engine.warmup(2);
                ResourceMonitor monitor = new ResourceMonitor(
                        cfg.getDouble("runtime.cpuLoadCap", 0.8),
                        cfg.getDouble("runtime.ramLoadCap", 0.8));
                for (String s : splits) {
                    Path imagesIn = raw.resolve(s);
                    Path csv = raw.resolve(s).resolve("_classes.csv");
                    Path imagesOut = yoloRoot.resolve(s).resolve("images");
                    Path labelsOut = yoloRoot.resolve(s).resolve("labels");
                    Files.createDirectories(imagesOut);
                    Files.createDirectories(labelsOut);
                    syncImagesSymlinkOrCopy(imagesIn, imagesOut);

                    CheckpointManager ckpt = new CheckpointManager(
                            Paths.get(cfg.getString("paths.checkpoints", "data/checkpoints"))
                                    .resolve("annot_" + s + ".json"),
                            "annot-" + s,
                            cfg.getInt("runtime.checkpointEveryN", 100));

                    AutoAnnotator.Config c = new AutoAnnotator.Config();
                    c.imagesDir = imagesIn;
                    c.labelsDir = labelsOut;
                    c.csvFile = csv;
                    c.maxImages = maxImages;
                    c.minConfidence = minConf;
                    new AutoAnnotator(engine, mapping, ckpt, monitor).run(c);
                    ckpt.close();
                }
            }

            DatasetValidator validator = new DatasetValidator(mapping.size());
            for (String s : splits) {
                DatasetValidator.Report r = validator.validateSplit(
                        yoloRoot.resolve(s).resolve("images"),
                        yoloRoot.resolve(s).resolve("labels"));
                LOG.info("Validation [{}] images={} matched={} missing={} ok={}",
                        s, r.images, r.matchedLabels, r.missingLabels, r.ok());
            }
            return 0;
        }

        private static void syncImagesSymlinkOrCopy(Path src, Path dst) throws java.io.IOException {
            try (var ds = Files.newDirectoryStream(src, "*.{jpg,jpeg,png}")) {
                for (Path p : ds) {
                    Path target = dst.resolve(p.getFileName());
                    if (Files.exists(target)) continue;
                    try {
                        Files.createSymbolicLink(target, p.toAbsolutePath());
                    } catch (Exception ignored) {
                        Files.copy(p, target);
                    }
                }
            }
        }
    }

    @Command(name = "infer-video", description = "Run inference on a single video (detection mode by default, optional FSM tracker via --track)")
    public static class InferVideo implements Callable<Integer> {
        @Option(names = {"--input", "-i"}, required = true) Path input;
        @Option(names = {"--output", "-o"}) Path output;
        @Option(names = "--method", defaultValue = "yolo") String method;
        @Option(names = "--max-frames", defaultValue = "-1") int maxFrames;
        @Option(names = "--frame-skip", defaultValue = "1") int frameSkip;
        @Option(names = "--track", description = "Optional FSM sequence (comma, e.g. 90,70,50). If absent, pure detection without tracker.") String trackSequence;
        @Option(names = "--report") Path report;
        @Option(names = "--no-cuda") boolean noCuda;
        @Option(names = "--conf", defaultValue = "0.30") float conf;

        @Override
        public Integer call() throws Exception {
            AppConfig cfg = AppConfig.load();
            ClassMapping mapping = cfg.classMapping();
            Path out = output != null ? output
                    : Paths.get(cfg.getString("paths.videosOut", "videos/output"))
                    .resolve(stripExt(input.getFileName().toString()) + "_" + method + ".mp4");
            Path rpt = report != null ? report
                    : Paths.get(cfg.getString("paths.reports", "reports"))
                    .resolve(stripExt(input.getFileName().toString()) + "_" + method + ".json");

            List<String> seq;
            if (trackSequence != null && !trackSequence.isBlank()) {
                seq = Arrays.asList(trackSequence.split("\\s*,\\s*"));
            } else {
                seq = List.of();
            }

            ResourceMonitor monitor = new ResourceMonitor(
                    cfg.getDouble("runtime.cpuLoadCap", 0.8),
                    cfg.getDouble("runtime.ramLoadCap", 0.8));
            CheckpointManager ckpt = new CheckpointManager(
                    Paths.get(cfg.getString("paths.checkpoints", "data/checkpoints"))
                            .resolve("video_" + stripExt(input.getFileName().toString()) + "_" + method + ".json"),
                    "video-" + input.getFileName(),
                    cfg.getInt("runtime.checkpointEveryN", 100));

            try (InferenceEngine engine = buildEngine(method, cfg, mapping, !noCuda)) {
                engine.warmup(2);
                VideoAnalyzer.Config c = new VideoAnalyzer.Config();
                c.input = input;
                c.output = out;
                c.reportJson = rpt;
                c.expectedSequence = seq;
                c.maxFrames = maxFrames;
                c.frameSkip = frameSkip;
                c.windowSize = cfg.getInt("tracker.windowFrames", 15);
                c.minDetections = cfg.getInt("tracker.minDetections", 5);
                c.minMeanConfidence = (float) cfg.getDouble("tracker.minMeanConfidence", 0.5);
                c.confidenceThreshold = conf;
                c.classFilter = classFilter(mapping, conf);
                VideoAnalyzer.Summary s = new VideoAnalyzer(engine, mapping, new FrameAnnotator(), ckpt, monitor)
                        .process(c);
                LOG.info("Video done frames={} fps={} transitions={}", s.framesProcessed,
                        String.format("%.2f", s.averageFps), s.transitions.size());
                ckpt.close();
            }
            return 0;
        }

        private static String stripExt(String n) {
            int dot = n.lastIndexOf('.');
            return dot < 0 ? n : n.substring(0, dot);
        }
    }

    @Command(name = "eval", description = "Evaluate an engine against a YOLO test split")
    public static class Evaluate implements Callable<Integer> {
        @Option(names = "--method", defaultValue = "yolo") String method;
        @Option(names = "--images", required = true) Path images;
        @Option(names = "--labels", required = true) Path labels;
        @Option(names = "--max-samples", defaultValue = "-1") int maxSamples;
        @Option(names = "--report") Path report;
        @Option(names = "--no-cuda") boolean noCuda;

        @Override
        public Integer call() throws Exception {
            AppConfig cfg = AppConfig.load();
            ClassMapping mapping = cfg.classMapping();
            try (InferenceEngine engine = buildEngine(method, cfg, mapping, !noCuda)) {
                engine.warmup(2);
                Evaluator.Report r = new Evaluator(engine, mapping).run(images, labels, maxSamples);
                Path rpt = report != null ? report
                        : Paths.get(cfg.getString("paths.reports", "reports"))
                        .resolve("eval_" + engine.name() + ".json");
                AtomicJsonWriter.write(rpt, r);
                LOG.info("Eval report written to {}", rpt);
            }
            return 0;
        }
    }

    @Command(name = "compare", description = "Compare engines on test split + videos")
    public static class Compare implements Callable<Integer> {
        @Option(names = "--test-images", required = true) Path testImages;
        @Option(names = "--test-labels", required = true) Path testLabels;
        @Option(names = "--max-samples", defaultValue = "200") int maxSamples;
        @Option(names = "--videos-dir") Path videosDir;
        @Option(names = "--max-frames", defaultValue = "-1") int maxFrames;
        @Option(names = "--no-cuda") boolean noCuda;
        @Option(names = "--report-md") Path reportMd;
        @Option(names = "--report-json") Path reportJson;
        @Option(names = "--report-csv") Path reportCsv;
        @Parameters(description = "Engines to compare (yolo, haarcnn)", arity = "1..*")
        List<String> engineNames;

        @Override
        public Integer call() throws Exception {
            AppConfig cfg = AppConfig.load();
            ClassMapping mapping = cfg.classMapping();
            List<InferenceEngine> engines = new ArrayList<>();
            try {
                for (String m : engineNames) engines.add(buildEngine(m, cfg, mapping, !noCuda));
                for (InferenceEngine e : engines) e.warmup(2);
                List<MethodComparator.VideoSpec> specs = new ArrayList<>();
                Path vDir = videosDir != null ? videosDir : Paths.get(cfg.getString("paths.videosIn", "videos/input"));
                Path outDir = Paths.get(cfg.getString("paths.videosOut", "videos/output"));
                Path reportsDir = Paths.get(cfg.getString("paths.reports", "reports"));
                List<String> seq1 = cfg.getStringList("tracker.video1Sequence");
                List<String> seq2 = cfg.getStringList("tracker.video2Sequence");
                if (seq1.isEmpty()) seq1 = List.of("90","70","50");
                if (seq2.isEmpty()) seq2 = List.of("110");
                addIfExists(vDir, "video1.wm", outDir, reportsDir, seq1, maxFrames, mapping, specs);
                addIfExists(vDir, "video2.wm", outDir, reportsDir, seq2, maxFrames, mapping, specs);
                addIfExists(vDir, "video1.wmv", outDir, reportsDir, seq1, maxFrames, mapping, specs);
                addIfExists(vDir, "video2.wmv", outDir, reportsDir, seq2, maxFrames, mapping, specs);

                MethodComparator.Report rep = new MethodComparator()
                        .compare(engines, mapping, testImages, testLabels, maxSamples, specs);

                Path md = reportMd != null ? reportMd : reportsDir.resolve("comparison.md");
                Path js = reportJson != null ? reportJson : reportsDir.resolve("comparison.json");
                Path cv = reportCsv != null ? reportCsv : reportsDir.resolve("comparison.csv");
                ReportWriter.writeMarkdown(md, rep);
                ReportWriter.writeJson(js, rep);
                ReportWriter.writeCsv(cv, rep);
                LOG.info("Comparison reports: md={} json={} csv={}", md, js, cv);
            } finally {
                for (InferenceEngine e : engines) try { e.close(); } catch (Exception ignored) {}
            }
            return 0;
        }

        private static void addIfExists(Path dir, String name, Path outDir, Path reportsDir,
                                        List<String> seq, int maxFrames, ClassMapping mapping,
                                        List<MethodComparator.VideoSpec> dst) {
            Path p = dir.resolve(name);
            if (!Files.exists(p)) return;
            String base = name.replaceAll("\\.[a-zA-Z0-9]+$", "");
            MethodComparator.VideoSpec s = MethodComparator.VideoSpec.of(
                    p,
                    outDir.resolve(base + "_compared.mp4"),
                    reportsDir.resolve(base + "_compared.json"),
                    seq).withFilter(mapping);
            s.maxFrames = maxFrames;
            dst.add(s);
        }
    }

    @Command(name = "bench", description = "FPS benchmark on a directory of images")
    public static class Bench implements Callable<Integer> {
        @Option(names = "--method", defaultValue = "yolo") String method;
        @Option(names = "--images", required = true) Path images;
        @Option(names = "--max-samples", defaultValue = "200") int maxSamples;
        @Option(names = "--no-cuda") boolean noCuda;

        @Override
        public Integer call() throws Exception {
            AppConfig cfg = AppConfig.load();
            ClassMapping mapping = cfg.classMapping();
            try (InferenceEngine engine = buildEngine(method, cfg, mapping, !noCuda)) {
                engine.warmup(5);
                FpsBenchmark.Result r = FpsBenchmark.runOnDirectory(engine, images, maxSamples);
                LOG.info("Bench done samples={} fps={}", r.samples, String.format("%.2f", r.fps));
            }
            return 0;
        }
    }

    @Command(name = "game", description = "3D driving sim using the fine-tuned YOLO model live")
    public static class Game implements Callable<Integer> {
        @Option(names = "--seed") Long seed;
        @Option(names = "--autonomous") boolean autonomous;
        @Option(names = "--no-cuda") boolean noCuda;
        @Option(names = "--conf", defaultValue = "0.20") float conf;
        @Option(names = "--segments", defaultValue = "30") int segments;
        @Option(names = "--width", defaultValue = "1280") int width;
        @Option(names = "--height", defaultValue = "720") int height;
        @Option(names = "--detect-width", defaultValue = "640") int detectWidth;
        @Option(names = "--detect-height", defaultValue = "360") int detectHeight;
        @Option(names = "--detect-period-ms", defaultValue = "100") long detectPeriodMs;

        @Override
        public Integer call() throws Exception {
            AppConfig appCfg = AppConfig.load();
            ClassMapping mapping = appCfg.classMapping();
            GameConfig gcfg = new GameConfig();
            if (seed != null) gcfg.seed = seed;
            gcfg.autonomous = autonomous;
            gcfg.detectionConfidence = conf;
            gcfg.segmentCount = segments;
            gcfg.width = width;
            gcfg.height = height;
            gcfg.detectionWidth = detectWidth;
            gcfg.detectionHeight = detectHeight;
            gcfg.detectionPeriodMs = detectPeriodMs;
            gcfg.useCuda = !noCuda;

            InferenceEngine engine = buildEngine("yolo", appCfg, mapping, !noCuda);
            engine.warmup(2);

            GameApp game = new GameApp(gcfg, engine);
            game.start();
            try { Thread.currentThread().join(); } catch (InterruptedException ignored) {}
            engine.close();
            return 0;
        }
    }

    @Command(name = "webcam", description = "Live detection from webcam in a window")
    public static class Webcam implements Callable<Integer> {
        @Option(names = "--device", defaultValue = "0") int device;
        @Option(names = "--width", defaultValue = "1280") int width;
        @Option(names = "--height", defaultValue = "720") int height;
        @Option(names = "--fps", defaultValue = "30") double fps;
        @Option(names = "--method", defaultValue = "yolo") String method;
        @Option(names = "--conf", defaultValue = "0.30") float conf;
        @Option(names = "--no-cuda") boolean noCuda;
        @Option(names = "--track", description = "Optional FSM sequence (comma, e.g. 90,70,50)") String trackSequence;

        @Override
        public Integer call() throws Exception {
            AppConfig cfg = AppConfig.load();
            ClassMapping mapping = cfg.classMapping();
            try (InferenceEngine engine = buildEngine(method, cfg, mapping, !noCuda)) {
                engine.warmup(2);
                WebcamAnalyzer.Config c = new WebcamAnalyzer.Config();
                c.deviceIndex = device;
                c.width = width;
                c.height = height;
                c.targetFps = fps;
                c.confidenceThreshold = conf;
                c.classFilter = classFilter(mapping, conf);
                if (trackSequence != null && !trackSequence.isBlank()) {
                    c.expectedSequence = Arrays.asList(trackSequence.split("\\s*,\\s*"));
                    c.useTracker = true;
                    c.windowSize = cfg.getInt("tracker.windowFrames", 15);
                    c.minDetections = cfg.getInt("tracker.minDetections", 5);
                    c.minMeanConfidence = (float) cfg.getDouble("tracker.minMeanConfidence", 0.5);
                }
                new WebcamAnalyzer(engine, mapping, new FrameAnnotator()).run(c);
            }
            return 0;
        }
    }
}
