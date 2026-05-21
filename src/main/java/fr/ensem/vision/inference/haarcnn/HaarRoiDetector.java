package fr.ensem.vision.inference.haarcnn;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.equalizeHist;

public final class HaarRoiDetector {

    private static final Logger LOG = LoggerFactory.getLogger(HaarRoiDetector.class);

    private final CascadeClassifier cascade;
    private final int minRoi;
    private final int maxRoi;
    private final double scaleFactor;
    private final int minNeighbors;

    public HaarRoiDetector(Path cascadePath, int minRoi, int maxRoi) {
        this.cascade = new CascadeClassifier(cascadePath.toAbsolutePath().toString());
        if (cascade.empty()) {
            throw new IllegalStateException("Failed to load Haar cascade: " + cascadePath);
        }
        this.minRoi = minRoi;
        this.maxRoi = maxRoi;
        this.scaleFactor = 1.1d;
        this.minNeighbors = 4;
        LOG.info("Haar cascade loaded from {}", cascadePath);
    }

    public List<Rect> detect(Mat bgr) {
        Mat gray = new Mat();
        cvtColor(bgr, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);
        RectVector rv = new RectVector();
        cascade.detectMultiScale(gray, rv, scaleFactor, minNeighbors, 0,
                new Size(minRoi, minRoi), new Size(maxRoi, maxRoi));
        List<Rect> out = new ArrayList<>((int) rv.size());
        for (long i = 0; i < rv.size(); i++) out.add(rv.get(i));
        gray.release();
        gray.close();
        return out;
    }

    public void close() {
        cascade.close();
    }
}
