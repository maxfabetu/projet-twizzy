package fr.ensem.vision.video;

import org.bytedeco.ffmpeg.global.avcodec;
import org.bytedeco.javacv.FFmpegFrameRecorder;
import org.bytedeco.javacv.FrameRecorder;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public final class VideoWriter implements AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(VideoWriter.class);

    private final FFmpegFrameRecorder recorder;
    private final OpenCVFrameConverter.ToMat converter;

    public VideoWriter(Path output, int width, int height, double fps) {
        try {
            if (output.getParent() != null) Files.createDirectories(output.getParent());
        } catch (IOException e) {
            throw new RuntimeException("Cannot create output dir", e);
        }
        this.recorder = new FFmpegFrameRecorder(output.toFile(), width, height);
        this.recorder.setFormat("mp4");
        this.recorder.setVideoCodec(avcodec.AV_CODEC_ID_H264);
        this.recorder.setFrameRate(fps);
        this.recorder.setVideoBitrate(4_000_000);
        this.recorder.setPixelFormat(org.bytedeco.ffmpeg.global.avutil.AV_PIX_FMT_YUV420P);
        try {
            recorder.start();
        } catch (FrameRecorder.Exception e) {
            throw new RuntimeException("Cannot start writer: " + output, e);
        }
        this.converter = new OpenCVFrameConverter.ToMat();
        LOG.info("Video writer opened: {} ({}x{} @ {} fps)", output, width, height, String.format("%.2f", fps));
    }

    public void write(Mat frame) {
        try {
            recorder.record(converter.convert(frame));
        } catch (FrameRecorder.Exception e) {
            throw new RuntimeException("Frame write failed", e);
        }
    }

    @Override
    public void close() {
        try {
            recorder.stop();
            recorder.close();
        } catch (FrameRecorder.Exception e) {
            LOG.warn("Writer close error: {}", e.getMessage());
        }
    }
}
