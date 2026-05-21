package fr.ensem.vision.data;

import fr.ensem.vision.detection.BoundingBox;
import fr.ensem.vision.detection.Detection;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public final class YoloLabelWriter {

    private YoloLabelWriter() {}

    public static void write(Path labelFile, List<Detection> detections, int imgW, int imgH) throws IOException {
        Path parent = labelFile.getParent();
        if (parent != null) Files.createDirectories(parent);
        List<String> lines = new ArrayList<>(detections.size());
        for (Detection d : detections) {
            BoundingBox b = d.getBox().clamp(imgW, imgH);
            float cx = b.getCenterX() / imgW;
            float cy = b.getCenterY() / imgH;
            float w = b.getWidth() / imgW;
            float h = b.getHeight() / imgH;
            if (w <= 0f || h <= 0f) continue;
            cx = clamp01(cx);
            cy = clamp01(cy);
            w = clamp01(w);
            h = clamp01(h);
            lines.add(String.format("%d %.6f %.6f %.6f %.6f",
                    d.getClassId(), cx, cy, w, h));
        }
        Files.write(labelFile, lines, StandardCharsets.UTF_8);
    }

    public static void writeClassesTxt(Path classesFile, List<String> names) throws IOException {
        Path parent = classesFile.getParent();
        if (parent != null) Files.createDirectories(parent);
        List<String> lines = new ArrayList<>(names.size());
        for (int i = 0; i < names.size(); i++) lines.add(i + " " + names.get(i));
        Files.write(classesFile, lines, StandardCharsets.UTF_8);
    }

    private static float clamp01(float v) {
        if (v < 0f) return 0f;
        if (v > 1f) return 1f;
        return v;
    }
}
