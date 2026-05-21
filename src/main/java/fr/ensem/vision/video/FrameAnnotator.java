package fr.ensem.vision.video;

import fr.ensem.vision.detection.BoundingBox;
import fr.ensem.vision.detection.Detection;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.bytedeco.opencv.global.opencv_imgproc.FONT_HERSHEY_SIMPLEX;
import static org.bytedeco.opencv.global.opencv_imgproc.LINE_AA;
import static org.bytedeco.opencv.global.opencv_imgproc.getTextSize;
import static org.bytedeco.opencv.global.opencv_imgproc.putText;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;

public final class FrameAnnotator {

    private static final Scalar GREEN = new Scalar(0, 200, 0, 0);
    private static final Scalar BLACK = new Scalar(0, 0, 0, 0);
    private static final Scalar WHITE = new Scalar(255, 255, 255, 0);

    private final Map<Integer, Scalar> classColors;

    public FrameAnnotator() {
        this.classColors = new HashMap<>();
        classColors.put(0, new Scalar(40, 220, 40, 0));
        classColors.put(1, new Scalar(40, 180, 240, 0));
        classColors.put(2, new Scalar(40, 80, 240, 0));
        classColors.put(3, new Scalar(220, 40, 220, 0));
    }

    public void drawDetections(Mat frame, List<Detection> detections) {
        for (Detection d : detections) {
            BoundingBox b = d.getBox();
            Scalar color = classColors.getOrDefault(d.getClassId(), GREEN);
            Rect r = new Rect((int) b.getX(), (int) b.getY(), (int) b.getWidth(), (int) b.getHeight());
            rectangle(frame, r, color, 2, LINE_AA, 0);
            String label = d.getLabel() + " " + String.format("%.0f", d.getConfidence() * 100f) + "%";
            drawLabel(frame, label, (int) b.getX(), (int) b.getY(), color);
            r.close();
        }
    }

    public void drawTrackerState(Mat frame, String state) {
        drawLabel(frame, "STATE: " + state, 10, 30, GREEN);
    }

    private void drawLabel(Mat frame, String text, int x, int y, Scalar bg) {
        int baseline[] = new int[] {0};
        Size ts = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.5, 1, baseline);
        int pad = 4;
        Rect bgRect = new Rect(x, Math.max(0, y - ts.height() - pad * 2), ts.width() + pad * 2, ts.height() + pad * 2);
        rectangle(frame, bgRect, bg, -1, LINE_AA, 0);
        Point textOrigin = new Point(x + pad, y - pad);
        putText(frame, text, textOrigin, FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, LINE_AA, false);
        bgRect.close();
        textOrigin.close();
        ts.close();
    }
}
