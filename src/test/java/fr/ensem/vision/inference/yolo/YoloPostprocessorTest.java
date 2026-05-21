package fr.ensem.vision.inference.yolo;

import fr.ensem.vision.config.ClassMapping;
import fr.ensem.vision.detection.Detection;
import org.bytedeco.opencv.opencv_core.Mat;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.CV_8UC3;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class YoloPostprocessorTest {

    @Test
    void decodesAttrsFirstLayoutAnchorsLast() {
        ClassMapping mapping = ClassMapping.defaultMapping();
        YoloPostprocessor pp = new YoloPostprocessor(mapping, 0.5f);

        int channels = 4 + mapping.size();
        int anchors = 3;
        float[] out = new float[channels * anchors];

        setAttrsFirst(out, channels, anchors, 0, 320f, 320f, 80f, 80f, new float[]{0.9f, 0.1f, 0.1f, 0.1f});
        setAttrsFirst(out, channels, anchors, 1, 100f, 100f, 50f, 50f, new float[]{0.1f, 0.85f, 0.1f, 0.1f});
        setAttrsFirst(out, channels, anchors, 2, 500f, 500f, 100f, 100f, new float[]{0.1f, 0.1f, 0.1f, 0.2f});

        Mat fake = new Mat(640, 640, CV_8UC3);
        LetterboxPreprocessor.Result letter = new LetterboxPreprocessor.Result(fake, 1f, 0, 0, 640, 640);

        List<Detection> dets = pp.decode(out, new long[]{1L, channels, anchors}, letter);
        assertEquals(2, dets.size());
        boolean has50 = false, has70 = false;
        for (Detection d : dets) {
            if ("50".equals(d.getLabel())) has50 = true;
            if ("70".equals(d.getLabel())) has70 = true;
        }
        assertTrue(has50);
        assertTrue(has70);
        fake.release();
        fake.close();
    }

    @Test
    void ignoresBelowThreshold() {
        ClassMapping mapping = ClassMapping.defaultMapping();
        YoloPostprocessor pp = new YoloPostprocessor(mapping, 0.9f);

        int channels = 4 + mapping.size();
        int anchors = 1;
        float[] out = new float[channels * anchors];
        setAttrsFirst(out, channels, anchors, 0, 320f, 320f, 80f, 80f, new float[]{0.5f, 0.1f, 0.1f, 0.1f});

        Mat fake = new Mat(640, 640, CV_8UC3);
        LetterboxPreprocessor.Result letter = new LetterboxPreprocessor.Result(fake, 1f, 0, 0, 640, 640);

        List<Detection> dets = pp.decode(out, new long[]{1L, channels, anchors}, letter);
        assertTrue(dets.isEmpty());
        fake.release();
        fake.close();
    }

    private static void setAttrsFirst(float[] out, int channels, int anchors, int anchorIdx,
                                      float cx, float cy, float w, float h, float[] clsScores) {
        out[0 * anchors + anchorIdx] = cx;
        out[1 * anchors + anchorIdx] = cy;
        out[2 * anchors + anchorIdx] = w;
        out[3 * anchors + anchorIdx] = h;
        for (int c = 0; c < clsScores.length; c++) {
            out[(4 + c) * anchors + anchorIdx] = clsScores[c];
        }
    }
}
