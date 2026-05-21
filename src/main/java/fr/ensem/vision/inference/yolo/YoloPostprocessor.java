package fr.ensem.vision.inference.yolo;

import fr.ensem.vision.config.ClassMapping;
import fr.ensem.vision.detection.BoundingBox;
import fr.ensem.vision.detection.Detection;

import java.util.ArrayList;
import java.util.List;

public final class YoloPostprocessor {

    private final ClassMapping mapping;
    private final float confThreshold;

    public YoloPostprocessor(ClassMapping mapping, float confThreshold) {
        this.mapping = mapping;
        this.confThreshold = confThreshold;
    }

    public List<Detection> decode(float[] output, long[] shape, LetterboxPreprocessor.Result letter) {
        if (shape.length != 3) throw new IllegalArgumentException("Expected 3D output");
        int channels = (int) shape[1];
        int anchors = (int) shape[2];

        int numYoloClasses = mapping.yoloNumClasses();
        boolean attrsFirst;
        if (channels == 4 + numYoloClasses) {
            attrsFirst = true;
        } else if (anchors == 4 + numYoloClasses) {
            attrsFirst = false;
            int tmp = channels;
            channels = anchors;
            anchors = tmp;
        } else if (channels > anchors) {
            attrsFirst = false;
            int tmp = channels;
            channels = anchors;
            anchors = tmp;
        } else {
            attrsFirst = true;
        }

        List<Detection> out = new ArrayList<>();
        for (int i = 0; i < anchors; i++) {
            float cx = at(output, channels, anchors, attrsFirst, 0, i);
            float cy = at(output, channels, anchors, attrsFirst, 1, i);
            float w = at(output, channels, anchors, attrsFirst, 2, i);
            float h = at(output, channels, anchors, attrsFirst, 3, i);

            int bestYoloCls = -1;
            float bestScore = 0f;
            for (int c = 4; c < channels; c++) {
                float s = at(output, channels, anchors, attrsFirst, c, i);
                if (s > bestScore) {
                    bestScore = s;
                    bestYoloCls = c - 4;
                }
            }
            if (bestYoloCls < 0 || bestScore < confThreshold) continue;
            ClassMapping.Entry entry = mapping.byYoloId(bestYoloCls);
            if (entry == null) continue;

            float[] orig = letter.unLetterbox(cx, cy, w, h);
            BoundingBox bb = BoundingBox.fromCxCyWh(orig[0], orig[1], orig[2], orig[3])
                    .clamp(letter.origW, letter.origH);
            out.add(new Detection(bb, entry.getId(), entry.getShortName(), bestScore));
        }
        return out;
    }

    private static float at(float[] arr, int channels, int anchors, boolean attrsFirst, int c, int a) {
        if (attrsFirst) return arr[c * anchors + a];
        return arr[a * channels + c];
    }
}
