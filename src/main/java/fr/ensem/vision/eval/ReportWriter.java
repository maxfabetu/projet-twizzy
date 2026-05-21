package fr.ensem.vision.eval;

import fr.ensem.vision.tracker.SequentialSignTracker;
import fr.ensem.vision.util.AtomicJsonWriter;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public final class ReportWriter {

    private ReportWriter() {}

    public static void writeJson(Path target, MethodComparator.Report report) throws IOException {
        AtomicJsonWriter.write(target, report);
    }

    public static void writeMarkdown(Path target, MethodComparator.Report report) throws IOException {
        Path parent = target.getParent();
        if (parent != null) Files.createDirectories(parent);
        List<String> lines = new ArrayList<>();
        lines.add("# Method Comparison Report");
        lines.add("");
        lines.add("Winner by mAP@0.5: **" + report.winnerByMap + "**");
        lines.add("Winner by FPS: **" + report.winnerByFps + "**");
        lines.add("");

        lines.add("## Detection metrics");
        lines.add("");
        lines.add("| Engine | mAP@0.5 | mAP@0.5:0.95 | Samples | Time (s) |");
        lines.add("|---|---|---|---|---|");
        for (MethodComparator.EngineComparison ec : report.byEngine.values()) {
            Evaluator.Report r = ec.evalReport;
            lines.add(String.format("| %s | %.3f | %.3f | %d | %.2f |",
                    ec.engineName, r.mapAt50, r.mapAt5095, r.totalSamples, r.seconds));
        }
        lines.add("");

        lines.add("## AP per class");
        lines.add("");
        for (MethodComparator.EngineComparison ec : report.byEngine.values()) {
            lines.add("### " + ec.engineName);
            lines.add("");
            lines.add("| Class | AP@0.5 | Precision | Recall | F1 | TP | FP | FN |");
            lines.add("|---|---|---|---|---|---|---|---|");
            for (Map.Entry<String, Double> e : ec.evalReport.apPerClass.entrySet()) {
                PrecisionRecallF1 prf = ec.evalReport.prf1PerClass.get(e.getKey());
                String prCells = prf == null ? "- | - | - | - | - | -" :
                        String.format("%.3f | %.3f | %.3f | %d | %d | %d",
                                prf.precision, prf.recall, prf.f1, prf.tp, prf.fp, prf.fn);
                lines.add(String.format("| %s | %.3f | %s |", e.getKey(), e.getValue(), prCells));
            }
            lines.add("");
        }

        lines.add("## Latency benchmark");
        lines.add("");
        lines.add("| Engine | FPS | mean (ms) | p50 (ms) | p95 (ms) | samples |");
        lines.add("|---|---|---|---|---|---|");
        for (MethodComparator.EngineComparison ec : report.byEngine.values()) {
            FpsBenchmark.Result b = ec.fpsBenchmark;
            if (b == null) continue;
            lines.add(String.format("| %s | %.2f | %.1f | %.1f | %.1f | %d |",
                    ec.engineName, b.fps, b.meanMs, b.p50Ms, b.p95Ms, b.samples));
        }
        lines.add("");

        lines.add("## Video outcomes");
        lines.add("");
        for (MethodComparator.EngineComparison ec : report.byEngine.values()) {
            lines.add("### " + ec.engineName);
            lines.add("");
            for (MethodComparator.VideoOutcome vo : ec.videoOutcomes) {
                lines.add("- **" + vo.videoFile + "**: expected=" + vo.expectedSequence
                        + " detected=" + vo.detectedSequence
                        + " match=" + vo.sequenceMatch
                        + " fps=" + String.format("%.2f", vo.averageFps));
                for (SequentialSignTracker.Transition t : vo.transitions) {
                    lines.add(String.format("  - step %d class %s frame %d t=%.0fms conf=%.2f support=%d",
                            t.stepIndex, t.expectedClass, t.frameIndex, t.timestampMs,
                            t.meanConfidence, t.supportingDetections));
                }
            }
            lines.add("");
        }

        Files.write(target, lines, StandardCharsets.UTF_8);
    }

    public static void writeCsv(Path target, MethodComparator.Report report) throws IOException {
        Path parent = target.getParent();
        if (parent != null) Files.createDirectories(parent);
        List<String> lines = new ArrayList<>();
        lines.add("engine,class,ap50,precision,recall,f1,tp,fp,fn");
        for (MethodComparator.EngineComparison ec : report.byEngine.values()) {
            for (Map.Entry<String, Double> e : ec.evalReport.apPerClass.entrySet()) {
                PrecisionRecallF1 prf = ec.evalReport.prf1PerClass.get(e.getKey());
                if (prf == null) {
                    lines.add(String.format("%s,%s,%.4f,,,,,,", ec.engineName, e.getKey(), e.getValue()));
                } else {
                    lines.add(String.format("%s,%s,%.4f,%.4f,%.4f,%.4f,%d,%d,%d",
                            ec.engineName, e.getKey(), e.getValue(),
                            prf.precision, prf.recall, prf.f1, prf.tp, prf.fp, prf.fn));
                }
            }
        }
        Files.write(target, lines, StandardCharsets.UTF_8);
    }
}
