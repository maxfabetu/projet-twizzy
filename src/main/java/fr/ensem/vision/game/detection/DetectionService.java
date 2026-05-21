package fr.ensem.vision.game.detection;

import com.jme3.post.SceneProcessor;
import com.jme3.profile.AppProfiler;
import com.jme3.renderer.RenderManager;
import com.jme3.renderer.Renderer;
import com.jme3.renderer.ViewPort;
import com.jme3.renderer.queue.RenderQueue;
import com.jme3.texture.FrameBuffer;
import com.jme3.util.BufferUtils;
import fr.ensem.vision.detection.Detection;
import fr.ensem.vision.inference.InferenceEngine;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

import static org.bytedeco.opencv.global.opencv_core.CV_8UC3;

public final class DetectionService implements SceneProcessor {

    private static final Logger LOG = LoggerFactory.getLogger(DetectionService.class);

    private final InferenceEngine engine;
    private final long periodMs;
    private final ExecutorService executor;
    private final AtomicReference<List<Detection>> latest = new AtomicReference<>(Collections.emptyList());
    private final AtomicBoolean inFlight = new AtomicBoolean(false);
    private final AtomicLong lastTriggerMs = new AtomicLong(0L);
    private final AtomicLong detectionCount = new AtomicLong(0L);
    private final AtomicReference<Long> lastInferMs = new AtomicReference<>(0L);

    private int width;
    private int height;
    private boolean initialized;
    private ByteBuffer cpuBuffer;

    public DetectionService(InferenceEngine engine, long periodMs) {
        this.engine = engine;
        this.periodMs = Math.max(20L, periodMs);
        this.executor = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r, "yolo-detection");
            t.setDaemon(true);
            return t;
        });
    }

    public List<Detection> getLatestDetections() {
        return latest.get();
    }

    public int getViewportWidth() { return width; }
    public int getViewportHeight() { return height; }

    public double getDetectionFps() {
        long ms = lastInferMs.get();
        return ms <= 0 ? 0d : 1000d / ms;
    }

    public long getDetectionCount() { return detectionCount.get(); }

    @Override
    public void initialize(RenderManager rm, ViewPort vp) {
        this.width = vp.getCamera().getWidth();
        this.height = vp.getCamera().getHeight();
        this.cpuBuffer = BufferUtils.createByteBuffer(width * height * 4);
        this.initialized = true;
    }

    @Override
    public void reshape(ViewPort vp, int w, int h) {
        this.width = w;
        this.height = h;
        this.cpuBuffer = BufferUtils.createByteBuffer(w * h * 4);
    }

    @Override
    public boolean isInitialized() {
        return initialized;
    }

    @Override
    public void preFrame(float tpf) {
    }

    @Override
    public void postQueue(RenderQueue rq) {
    }

    @Override
    public void postFrame(FrameBuffer out) {
        if (!initialized) return;
        long now = System.currentTimeMillis();
        if (now - lastTriggerMs.get() < periodMs) return;
        if (!inFlight.compareAndSet(false, true)) return;
        try {
            cpuBuffer.clear();
            Renderer renderer = RenderManagerHolder.RENDERER;
            if (renderer == null) {
                inFlight.set(false);
                return;
            }
            renderer.readFrameBuffer(out, cpuBuffer);
        } catch (Throwable t) {
            inFlight.set(false);
            LOG.warn("readFrameBuffer failed: {}", t.getMessage());
            return;
        }
        lastTriggerMs.set(now);
        final int w = width;
        final int h = height;
        final byte[] copy = new byte[w * h * 4];
        cpuBuffer.position(0);
        cpuBuffer.get(copy);
        executor.submit(() -> runInference(copy, w, h));
    }

    @Override
    public void cleanup() {
        executor.shutdownNow();
    }

    @Override
    public void setProfiler(AppProfiler profiler) {
    }

    private void runInference(byte[] rgba, int w, int h) {
        try {
            long t0 = System.currentTimeMillis();
            Mat bgr = new Mat(h, w, CV_8UC3);
            org.bytedeco.javacpp.BytePointer ptr = bgr.data();
            byte[] dst = new byte[w * h * 3];
            int stride = w * 4;
            for (int y = 0; y < h; y++) {
                int srcY = (h - 1 - y) * stride;
                int dstY = y * w * 3;
                for (int x = 0; x < w; x++) {
                    int s = srcY + x * 4;
                    int d = dstY + x * 3;
                    byte r = rgba[s];
                    byte g = rgba[s + 1];
                    byte b = rgba[s + 2];
                    dst[d] = b;
                    dst[d + 1] = g;
                    dst[d + 2] = r;
                }
            }
            ptr.put(dst, 0, dst.length);
            List<Detection> detections = engine.infer(bgr);
            latest.set(detections == null ? Collections.emptyList() : detections);
            detectionCount.incrementAndGet();
            lastInferMs.set(System.currentTimeMillis() - t0);
            bgr.release();
            bgr.close();
        } catch (Throwable t) {
            LOG.warn("Inference error: {}", t.getMessage());
        } finally {
            inFlight.set(false);
        }
    }

    public static void setRenderer(Renderer r) {
        RenderManagerHolder.RENDERER = r;
    }

    private static final class RenderManagerHolder {
        static volatile Renderer RENDERER;
    }
}
