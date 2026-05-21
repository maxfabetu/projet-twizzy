package fr.ensem.vision.inference.haarcnn;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import fr.ensem.vision.inference.ModelLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.Collections;
import java.util.Map;

public final class CnnClassifier implements AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(CnnClassifier.class);

    public static final class Result {
        public final int classId;
        public final float confidence;
        public final float[] probabilities;

        public Result(int classId, float confidence, float[] probabilities) {
            this.classId = classId;
            this.confidence = confidence;
            this.probabilities = probabilities;
        }
    }

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;
    private final int inputSize;
    private final int numClasses;
    private final long[] inputShape;

    public CnnClassifier(Path modelPath, int inputSize, int numClasses, boolean useCuda) throws OrtException {
        this.env = ModelLoader.environment();
        this.session = ModelLoader.loadOnnx(env, modelPath, useCuda);
        this.inputName = session.getInputNames().iterator().next();
        this.inputSize = inputSize;
        this.numClasses = numClasses;
        this.inputShape = new long[] { 1L, 3L, inputSize, inputSize };
        LOG.info("CNN classifier loaded: input={} size={} classes={}", inputName, inputSize, numClasses);
    }

    public synchronized Result classify(FloatBuffer chw) {
        try (OnnxTensor in = OnnxTensor.createTensor(env, chw, inputShape)) {
            Map<String, OnnxTensor> inputs = Collections.singletonMap(inputName, in);
            try (OrtSession.Result res = session.run(inputs)) {
                Object raw = res.get(0).getValue();
                float[] logits = flatten(raw);
                float[] probs = softmax(logits);
                int best = 0;
                for (int i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
                return new Result(best, probs[best], probs);
            }
        } catch (OrtException e) {
            throw new RuntimeException("CNN classify failed", e);
        }
    }

    private static float[] flatten(Object raw) {
        if (raw instanceof float[]) return (float[]) raw;
        if (raw instanceof float[][]) return ((float[][]) raw)[0];
        throw new IllegalStateException("Unsupported CNN output: " + raw.getClass());
    }

    private static float[] softmax(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) if (v > max) max = v;
        float sum = 0f;
        float[] e = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            e[i] = (float) Math.exp(logits[i] - max);
            sum += e[i];
        }
        if (sum <= 0f) sum = 1f;
        for (int i = 0; i < e.length; i++) e[i] /= sum;
        return e;
    }

    public int inputSize() { return inputSize; }
    public int numClasses() { return numClasses; }

    @Override
    public void close() {
        try {
            session.close();
        } catch (OrtException e) {
            LOG.warn("CNN session close failed: {}", e.getMessage());
        }
    }
}
