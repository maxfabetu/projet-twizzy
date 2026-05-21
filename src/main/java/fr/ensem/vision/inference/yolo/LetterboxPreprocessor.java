package fr.ensem.vision.inference.yolo;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;

import static org.bytedeco.opencv.global.opencv_core.BORDER_CONSTANT;
import static org.bytedeco.opencv.global.opencv_core.copyMakeBorder;
import static org.bytedeco.opencv.global.opencv_imgproc.INTER_LINEAR;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

public final class LetterboxPreprocessor {

    public static final class Result {
        public final Mat letterboxed;
        public final float ratio;
        public final int padLeft;
        public final int padTop;
        public final int origW;
        public final int origH;

        public Result(Mat letterboxed, float ratio, int padLeft, int padTop, int origW, int origH) {
            this.letterboxed = letterboxed;
            this.ratio = ratio;
            this.padLeft = padLeft;
            this.padTop = padTop;
            this.origW = origW;
            this.origH = origH;
        }

        public float[] unLetterbox(float cx, float cy, float w, float h) {
            float fx = (cx - padLeft) / ratio;
            float fy = (cy - padTop) / ratio;
            float fw = w / ratio;
            float fh = h / ratio;
            return new float[] { fx, fy, fw, fh };
        }
    }

    private final int target;
    private final Scalar padColor;

    public LetterboxPreprocessor(int target) {
        this.target = target;
        this.padColor = new Scalar(114, 114, 114, 0);
    }

    public Result process(Mat bgr) {
        int origW = bgr.cols();
        int origH = bgr.rows();
        float ratio = Math.min(target / (float) origW, target / (float) origH);
        int newW = Math.round(origW * ratio);
        int newH = Math.round(origH * ratio);

        Mat resized = new Mat();
        resize(bgr, resized, new Size(newW, newH), 0, 0, INTER_LINEAR);

        int padW = target - newW;
        int padH = target - newH;
        int padLeft = padW / 2;
        int padRight = padW - padLeft;
        int padTop = padH / 2;
        int padBottom = padH - padTop;

        Mat out = new Mat();
        copyMakeBorder(resized, out, padTop, padBottom, padLeft, padRight, BORDER_CONSTANT, padColor);

        resized.release();
        resized.close();
        return new Result(out, ratio, padLeft, padTop, origW, origH);
    }

}
