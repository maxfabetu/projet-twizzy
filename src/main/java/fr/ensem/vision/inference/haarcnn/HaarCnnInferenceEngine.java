package fr.ensem.vision.inference.haarcnn;

import ai.onnxruntime.OrtException;
import fr.ensem.vision.config.ClassMapping;
import fr.ensem.vision.detection.BoundingBox;
import fr.ensem.vision.detection.Detection;
import fr.ensem.vision.inference.InferenceEngine;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public final class HaarCnnInferenceEngine implements InferenceEngine {

    private static final Logger LOG = LoggerFactory.getLogger(HaarCnnInferenceEngine.class);

    private final HaarRoiDetector haar;
    private final CnnClassifier cnn;
    private final RoiPreprocessor preprocessor;
    private final ClassMapping mapping;
    private final float confThreshold;

    public HaarCnnInferenceEngine(Path cascadePath, Path cnnPath, ClassMapping mapping,
                                  int cnnInputSize, int minRoi, int maxRoi,
                                  float confThreshold, boolean useCuda) throws OrtException {
        this.haar = new HaarRoiDetector(cascadePath, minRoi, maxRoi);
        this.cnn = new CnnClassifier(cnnPath, cnnInputSize, mapping.size(), useCuda);
        this.preprocessor = new RoiPreprocessor(cnnInputSize);
        this.mapping = mapping;
        this.confThreshold = confThreshold;
    }

    @Override
    public synchronized List<Detection> infer(Mat frame) {
        List<Rect> rois = haar.detect(frame);
        List<Detection> out = new ArrayList<>(rois.size());
        for (Rect r : rois) {
            FloatBuffer tensor = preprocessor.toTensor(frame, r);
            CnnClassifier.Result cls = cnn.classify(tensor);
            if (cls.confidence < confThreshold) continue;
            BoundingBox bb = new BoundingBox(r.x(), r.y(), r.width(), r.height());
            String label = mapping.labelForId(cls.classId);
            out.add(new Detection(bb, cls.classId, label, cls.confidence));
        }
        return out;
    }

    @Override
    public String name() { return "haar-cnn"; }

    @Override
    public void warmup(int iterations) {
        LOG.info("Haar+CNN warmup: {} iters (no-op preload)", iterations);
    }

    @Override
    public void close() {
        haar.close();
        cnn.close();
    }
}
