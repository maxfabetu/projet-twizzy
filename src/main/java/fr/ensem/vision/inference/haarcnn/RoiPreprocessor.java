package fr.ensem.vision.inference.haarcnn;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Size;

import java.nio.FloatBuffer;

import static org.bytedeco.opencv.global.opencv_core.CV_32FC3;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2RGB;
import static org.bytedeco.opencv.global.opencv_imgproc.INTER_LINEAR;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

public final class RoiPreprocessor {

    private final int targetSize;

    public RoiPreprocessor(int targetSize) {
        this.targetSize = targetSize;
    }

    public FloatBuffer toTensor(Mat bgr, Rect roi) {
        Mat crop = new Mat(bgr, roi);
        Mat resized = new Mat();
        resize(crop, resized, new Size(targetSize, targetSize), 0, 0, INTER_LINEAR);
        Mat rgb = new Mat();
        cvtColor(resized, rgb, COLOR_BGR2RGB);
        Mat f32 = new Mat();
        rgb.convertTo(f32, CV_32FC3, 1.0d / 255.0d, 0.0d);

        int chans = f32.channels();
        int h = f32.rows();
        int w = f32.cols();
        FloatPointer fp = new FloatPointer(f32.ptr());
        int hwc = chans * h * w;
        float[] hwcArr = new float[hwc];
        fp.get(hwcArr, 0, hwc);

        FloatBuffer dst = FloatBuffer.allocate(hwc);
        int planeSize = h * w;
        for (int c = 0; c < chans; c++) {
            for (int i = 0; i < planeSize; i++) {
                dst.put(c * planeSize + i, hwcArr[i * chans + c]);
            }
        }
        dst.position(0).limit(hwc);

        crop.release(); crop.close();
        resized.release(); resized.close();
        rgb.release(); rgb.close();
        f32.release(); f32.close();
        fp.deallocate();
        return dst;
    }

    public int targetSize() { return targetSize; }
}
