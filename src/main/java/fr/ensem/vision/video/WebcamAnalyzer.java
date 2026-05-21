package fr.ensem.vision.video;

import fr.ensem.vision.config.ClassMapping;
import fr.ensem.vision.detection.Detection;
import fr.ensem.vision.detection.DetectionFilter;
import fr.ensem.vision.inference.InferenceEngine;
import fr.ensem.vision.tracker.DetectionAggregator;
import fr.ensem.vision.tracker.SequentialSignTracker;
import fr.ensem.vision.tracker.TemporalFilter;
import fr.ensem.vision.util.EtaCalculator;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.WindowConstants;
import java.awt.event.KeyEvent;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_imgproc.FONT_HERSHEY_SIMPLEX;
import static org.bytedeco.opencv.global.opencv_imgproc.LINE_AA;
import static org.bytedeco.opencv.global.opencv_imgproc.putText;

public final class WebcamAnalyzer {

    private static final Logger LOG = LoggerFactory.getLogger(WebcamAnalyzer.class);

    public static final class Config {
        public int deviceIndex = 0;
        public int width = 1280;
        public int height = 720;
        public double targetFps = 30.0;
        public float confidenceThreshold = 0.30f;
        public List<String> expectedSequence;
        public int windowSize = 15;
        public int minDetections = 5;
        public float minMeanConfidence = 0.5f;
        public DetectionFilter classFilter;
        public boolean useTracker = false;
        public String windowTitle = "Vision ENSEM - Webcam live";
    }

    private final InferenceEngine engine;
    private final ClassMapping mapping;
    private final FrameAnnotator annotator;

    public WebcamAnalyzer(InferenceEngine engine, ClassMapping mapping, FrameAnnotator annotator) {
        this.engine = engine;
        this.mapping = mapping;
        this.annotator = annotator;
    }

    public void run(Config cfg) {
        OpenCVFrameGrabber grabber = new OpenCVFrameGrabber(cfg.deviceIndex);
        grabber.setImageWidth(cfg.width);
        grabber.setImageHeight(cfg.height);
        grabber.setFrameRate(cfg.targetFps);
        try {
            grabber.start();
        } catch (FrameGrabber.Exception e) {
            throw new RuntimeException("Cannot open webcam (device " + cfg.deviceIndex + ")", e);
        }
        int realW = grabber.getImageWidth();
        int realH = grabber.getImageHeight();
        LOG.info("Webcam opened: device={} {}x{} @ {} fps requested",
                cfg.deviceIndex, realW, realH, cfg.targetFps);

        CanvasFrame canvas = new CanvasFrame(cfg.windowTitle, CanvasFrame.getDefaultGamma() / 2.2);
        canvas.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        canvas.setCanvasSize(realW, realH);

        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

        DetectionAggregator agg = cfg.useTracker ? new DetectionAggregator(cfg.windowSize) : null;
        TemporalFilter tf = cfg.useTracker ? new TemporalFilter(cfg.minDetections, cfg.minMeanConfidence) : null;
        SequentialSignTracker tracker = (cfg.useTracker && cfg.expectedSequence != null && !cfg.expectedSequence.isEmpty())
                ? new SequentialSignTracker(cfg.expectedSequence, mapping, agg, tf) : null;

        EtaCalculator eta = new EtaCalculator(20);
        long startNs = System.nanoTime();
        long frameIdx = 0;

        try {
            while (canvas.isVisible()) {
                KeyEvent kev = null;
                try { kev = canvas.waitKey(1); } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    break;
                }
                if (kev != null && kev.getKeyCode() == KeyEvent.VK_ESCAPE) break;

                Frame raw = grabber.grab();
                if (raw == null) continue;
                Mat frame;
                try (PointerScope scope = new PointerScope()) {
                    frame = converter.convertToMat(raw);
                    if (frame == null || frame.empty()) continue;
                    Mat work = frame.clone();

                    long t0 = System.nanoTime();
                    List<Detection> detections = engine.infer(work);
                    if (cfg.classFilter != null) detections = cfg.classFilter.apply(detections);

                    if (tracker != null) {
                        List<Detection> withFrame = new ArrayList<>(detections.size());
                        double tsMs = (frameIdx * 1000.0) / Math.max(1.0, cfg.targetFps);
                        for (Detection d : detections) withFrame.add(d.withFrame(frameIdx, tsMs));
                        tracker.update(frameIdx, tsMs, withFrame);
                    }

                    annotator.drawDetections(work, detections);
                    if (tracker != null) annotator.drawTrackerState(work, tracker.currentStateLabel());

                    long inferMs = (System.nanoTime() - t0) / 1_000_000L;
                    eta.tick();
                    double fps = eta.fps();
                    drawHud(work, fps, inferMs, detections.size(), tracker);

                    canvas.showImage(converter.convert(work));
                    work.release();
                    work.close();
                }

                frameIdx++;
                if (frameIdx % 30 == 0) {
                    LOG.info("Webcam frame {} fps={} infer={}detections",
                            frameIdx, String.format("%.1f", eta.fps()), 0);
                }
            }
        } catch (FrameGrabber.Exception e) {
            LOG.error("Webcam grab error: {}", e.getMessage());
        } finally {
            try { grabber.stop(); grabber.close(); } catch (Exception ignored) {}
            canvas.dispose();
            long elapsed = System.nanoTime() - startNs;
            LOG.info("Webcam closed: processed={} frames in {}s avg={}fps",
                    frameIdx, String.format("%.1f", elapsed / 1e9),
                    String.format("%.1f", frameIdx / (elapsed / 1e9)));
        }
    }

    private static void drawHud(Mat frame, double fps, long inferMs, int nDets, SequentialSignTracker tracker) {
        String hud = String.format("FPS:%.1f  infer:%dms  dets:%d  ESC=quit", fps, inferMs, nDets);
        Point p = new Point(10, frame.rows() - 12);
        putText(frame, hud, p, FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 255, 0, 0), 1, LINE_AA, false);
        p.close();
    }
}
