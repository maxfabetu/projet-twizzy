package fr.ensem.vision.util;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;

import java.nio.FloatBuffer;

import static org.bytedeco.opencv.global.opencv_core.CV_32FC3;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2RGB;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

public final class MatTensorConverter {

    private MatTensorConverter() {}

    public static FloatBuffer matToChwFloat(Mat bgr, int targetSize, FloatBuffer dst) {
        Mat resized = new Mat();
        resize(bgr, resized, new Size(targetSize, targetSize));
        Mat rgb = new Mat();
        cvtColor(resized, rgb, COLOR_BGR2RGB);
        Mat f32 = new Mat();
        rgb.convertTo(f32, CV_32FC3, 1.0d / 255.0d, 0.0d);

        int h = f32.rows();
        int w = f32.cols();
        int chans = f32.channels();
        FloatPointer fp = new FloatPointer(f32.ptr());
        int hwc = h * w * chans;
        float[] hwcArr = new float[hwc];
        fp.get(hwcArr, 0, hwc);

        if (dst == null || dst.capacity() < hwc) dst = FloatBuffer.allocate(hwc);
        dst.clear();
        int planeSize = h * w;
        for (int c = 0; c < chans; c++) {
            for (int i = 0; i < planeSize; i++) {
                dst.put(c * planeSize + i, hwcArr[i * chans + c]);
            }
        }
        dst.position(0).limit(hwc);

        resized.release(); resized.close();
        rgb.release(); rgb.close();
        f32.release(); f32.close();
        fp.deallocate();
        return dst;
    }

    public static FloatBuffer matLetterboxedToChw(Mat letterboxed, FloatBuffer dst) {
        int h = letterboxed.rows();
        int w = letterboxed.cols();
        Mat rgb = new Mat();
        cvtColor(letterboxed, rgb, COLOR_BGR2RGB);
        Mat f32 = new Mat();
        rgb.convertTo(f32, CV_32FC3, 1.0d / 255.0d, 0.0d);

        int chans = f32.channels();
        FloatPointer fp = new FloatPointer(f32.ptr());
        int hwc = h * w * chans;
        float[] hwcArr = new float[hwc];
        fp.get(hwcArr, 0, hwc);

        if (dst == null || dst.capacity() < hwc) dst = FloatBuffer.allocate(hwc);
        dst.clear();
        int planeSize = h * w;
        for (int c = 0; c < chans; c++) {
            for (int i = 0; i < planeSize; i++) {
                dst.put(c * planeSize + i, hwcArr[i * chans + c]);
            }
        }
        dst.position(0).limit(hwc);

        rgb.release(); rgb.close();
        f32.release(); f32.close();
        fp.deallocate();
        return dst;
    }
}
