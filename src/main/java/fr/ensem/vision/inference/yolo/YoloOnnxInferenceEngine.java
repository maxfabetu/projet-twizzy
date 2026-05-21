package fr.ensem.vision.inference.yolo;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import fr.ensem.vision.config.ClassMapping;
import fr.ensem.vision.detection.Detection;
import fr.ensem.vision.inference.InferenceEngine;
import fr.ensem.vision.inference.ModelLoader;
import fr.ensem.vision.util.MatTensorConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class YoloOnnxInferenceEngine implements InferenceEngine {

    private static final Logger LOG = LoggerFactory.getLogger(YoloOnnxInferenceEngine.class);

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;
    private final int inputSize;
    private final LetterboxPreprocessor preprocessor;
    private final YoloPostprocessor postprocessor;
    private final NonMaxSuppression nms;
    private final long[] inputShape;

    public YoloOnnxInferenceEngine(Path modelPath, ClassMapping mapping,
                                   int inputSize, float confThreshold,
                                   float iouThreshold, boolean useCuda) throws OrtException {
        this.env = ModelLoader.environment();
        this.session = ModelLoader.loadOnnx(env, modelPath, useCuda);
        this.inputName = session.getInputNames().iterator().next();
        this.inputSize = inputSize;
        this.preprocessor = new LetterboxPreprocessor(inputSize);
        this.postprocessor = new YoloPostprocessor(mapping, confThreshold);
        this.nms = new NonMaxSuppression(iouThreshold);
        this.inputShape = new long[] { 1L, 3L, inputSize, inputSize };
        LOG.info("YOLOv8 ONNX loaded: input={} size={}x{} cuda={}", inputName, inputSize, inputSize, useCuda);
    }

    @Override
    public synchronized List<Detection> infer(Mat frame) {
        LetterboxPreprocessor.Result lr = preprocessor.process(frame);
        try {
            FloatBuffer fb = MatTensorConverter.matLetterboxedToChw(lr.letterboxed, null);
            try (OnnxTensor in = OnnxTensor.createTensor(env, fb, inputShape)) {
                Map<String, OnnxTensor> inputs = new HashMap<>();
                inputs.put(inputName, in);
                try (OrtSession.Result res = session.run(inputs)) {
                    Object raw = res.get(0).getValue();
                    float[] flat = flattenTo1d(raw);
                    long[] shape = ((ai.onnxruntime.OnnxTensor) res.get(0)).getInfo().getShape();
                    List<Detection> decoded = postprocessor.decode(flat, shape, lr);
                    return nms.apply(decoded);
                }
            }
        } catch (OrtException e) {
            throw new RuntimeException("YOLO inference failed", e);
        } finally {
            lr.letterboxed.release();
            lr.letterboxed.close();
        }
    }

    private static float[] flattenTo1d(Object raw) {
        if (raw instanceof float[]) return (float[]) raw;
        if (raw instanceof float[][]) {
            float[][] a = (float[][]) raw;
            float[] out = new float[a.length * a[0].length];
            int k = 0;
            for (float[] r : a) { System.arraycopy(r, 0, out, k, r.length); k += r.length; }
            return out;
        }
        if (raw instanceof float[][][]) {
            float[][][] a = (float[][][]) raw;
            int d0 = a.length, d1 = a[0].length, d2 = a[0][0].length;
            float[] out = new float[d0 * d1 * d2];
            int k = 0;
            for (int i = 0; i < d0; i++)
                for (int j = 0; j < d1; j++) {
                    System.arraycopy(a[i][j], 0, out, k, d2);
                    k += d2;
                }
            return out;
        }
        throw new IllegalStateException("Unsupported ONNX output type: " + raw.getClass());
    }

    @Override
    public String name() { return "yolov8-onnx"; }

    @Override
    public void warmup(int iterations) {
        try {
            int n = Math.max(1, iterations);
            for (int i = 0; i < n; i++) {
                FloatBuffer fb = FloatBuffer.allocate(3 * inputSize * inputSize);
                try (OnnxTensor in = OnnxTensor.createTensor(env, fb, inputShape)) {
                    Map<String, OnnxTensor> inputs = Collections.singletonMap(inputName, in);
                    try (OrtSession.Result res = session.run(inputs)) {
                        res.get(0).getValue();
                    }
                }
            }
            LOG.info("YOLO warmup complete ({} iters)", n);
        } catch (OrtException e) {
            LOG.warn("Warmup failed: {}", e.getMessage());
        }
    }

    @Override
    public void close() {
        try {
            if (session != null) session.close();
        } catch (OrtException e) {
            LOG.warn("Session close error: {}", e.getMessage());
        }
    }
}
