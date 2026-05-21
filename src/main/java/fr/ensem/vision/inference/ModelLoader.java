package fr.ensem.vision.inference;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtLoggingLevel;
import ai.onnxruntime.OrtSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;

public final class ModelLoader {

    private static final Logger LOG = LoggerFactory.getLogger(ModelLoader.class);

    private ModelLoader() {}

    public static OrtSession loadOnnx(OrtEnvironment env, Path modelPath, boolean useCuda) throws OrtException {
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        opts.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING);
        opts.setInterOpNumThreads(1);
        opts.setIntraOpNumThreads(Math.max(1, Runtime.getRuntime().availableProcessors() / 2));
        if (useCuda) {
            try {
                opts.addCUDA(0);
                LOG.info("CUDA provider enabled for {}", modelPath);
            } catch (Throwable t) {
                LOG.warn("CUDA provider not available, falling back to CPU: {}", t.getMessage());
            }
        }
        return env.createSession(modelPath.toAbsolutePath().toString(), opts);
    }

    public static OrtEnvironment environment() {
        return OrtEnvironment.getEnvironment();
    }
}
