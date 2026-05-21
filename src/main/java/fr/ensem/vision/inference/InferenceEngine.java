package fr.ensem.vision.inference;

import fr.ensem.vision.detection.Detection;
import org.bytedeco.opencv.opencv_core.Mat;

import java.util.List;

public interface InferenceEngine extends AutoCloseable {

    List<Detection> infer(Mat frame);

    String name();

    void warmup(int iterations);

    @Override
    void close();
}
