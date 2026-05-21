package fr.ensem.vision.cli;

import picocli.CommandLine.Option;

import java.nio.file.Path;
import java.nio.file.Paths;

public final class CliArgs {

    @Option(names = {"--input", "-i"}, description = "Input video or directory")
    public Path input;

    @Option(names = {"--output", "-o"}, description = "Output path")
    public Path output;

    @Option(names = {"--method", "-m"}, description = "Inference method: yolo | haarcnn", defaultValue = "yolo")
    public String method = "yolo";

    @Option(names = {"--max-frames"}, description = "Max frames to process (smoke test)")
    public int maxFrames = -1;

    @Option(names = {"--max-images"}, description = "Max images to process (smoke test)")
    public int maxImages = -1;

    @Option(names = {"--threads"}, description = "Threads (defaults to 80% of CPUs)")
    public int threads = -1;

    @Option(names = {"--checkpoint"}, description = "Checkpoint file", defaultValue = "data/checkpoints/run.json")
    public Path checkpoint = Paths.get("data/checkpoints/run.json");

    @Option(names = {"--resume"}, description = "Resume from checkpoint")
    public boolean resume = false;

    @Option(names = {"--conf"}, description = "Confidence threshold override")
    public Float confidence;

    @Option(names = {"--iou"}, description = "NMS IoU threshold override")
    public Float iou;

    @Option(names = {"--cuda"}, description = "Force CUDA on/off", defaultValue = "true")
    public boolean cuda = true;

    @Option(names = {"--no-cuda"}, description = "Disable CUDA")
    public boolean noCuda = false;

    @Option(names = {"--frame-skip"}, description = "Process 1 frame out of N", defaultValue = "1")
    public int frameSkip = 1;

    @Option(names = {"--report"}, description = "Report output path")
    public Path report;

    @Option(names = {"--test-set"}, description = "Test set root directory")
    public Path testSet;

    @Option(names = {"--videos"}, description = "Videos directory for comparison")
    public Path videosDir;

    @Option(names = {"--config"}, description = "Override config path")
    public Path configPath;

    @Option(names = {"--video1-sequence"}, description = "Expected sequence on video1, comma separated", defaultValue = "90,70,50")
    public String video1Sequence = "90,70,50";

    @Option(names = {"--video2-sequence"}, description = "Expected sequence on video2, comma separated", defaultValue = "110")
    public String video2Sequence = "110";

    @Option(names = {"--help", "-h"}, usageHelp = true, description = "Show help")
    public boolean help;

    public boolean useCuda() {
        return cuda && !noCuda;
    }
}
