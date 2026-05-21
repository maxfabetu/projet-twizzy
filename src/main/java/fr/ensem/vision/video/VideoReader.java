package fr.ensem.vision.video;

import org.bytedeco.ffmpeg.global.avutil;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;

public final class VideoReader implements AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(VideoReader.class);

    private final FFmpegFrameGrabber grabber;
    private final OpenCVFrameConverter.ToMat converter;
    private final int width;
    private final int height;
    private final double fps;
    private final long totalFrames;
    private long frameIndex = 0L;

    public VideoReader(Path videoPath) {
        this.grabber = new FFmpegFrameGrabber(videoPath.toFile());
        this.grabber.setPixelFormat(avutil.AV_PIX_FMT_BGR24);
        try {
            grabber.start();
        } catch (FrameGrabber.Exception e) {
            throw new RuntimeException("Cannot open video: " + videoPath, e);
        }
        this.converter = new OpenCVFrameConverter.ToMat();
        this.width = grabber.getImageWidth();
        this.height = grabber.getImageHeight();
        this.fps = grabber.getFrameRate() > 0 ? grabber.getFrameRate() : 30d;
        this.totalFrames = grabber.getLengthInFrames();
        LOG.info("Video opened: {} ({}x{} @ {} fps, ~{} frames)", videoPath, width, height,
                String.format("%.2f", fps), totalFrames);
    }

    public Mat readFrame() {
        try {
            Frame f;
            while ((f = grabber.grabImage()) != null) {
                Mat m = converter.convert(f);
                if (m == null || m.empty()) continue;
                frameIndex++;
                return m.clone();
            }
            return null;
        } catch (FrameGrabber.Exception e) {
            throw new RuntimeException("Frame grab failed", e);
        }
    }

    public int width() { return width; }
    public int height() { return height; }
    public double fps() { return fps; }
    public long totalFrames() { return totalFrames; }
    public long currentFrameIndex() { return frameIndex; }
    public double currentTimestampMs() { return (frameIndex - 1) * 1000d / fps; }

    @Override
    public void close() {
        try {
            grabber.stop();
            grabber.close();
        } catch (FrameGrabber.Exception e) {
            LOG.warn("Reader close error: {}", e.getMessage());
        }
    }
}
