package fr.ensem.vision.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public final class DatasetValidator {

    private static final Logger LOG = LoggerFactory.getLogger(DatasetValidator.class);

    public static final class Report {
        public int images;
        public int matchedLabels;
        public int missingLabels;
        public int malformedLines;
        public int outOfRange;
        public int unknownClass;
        public List<String> errors = new ArrayList<>();

        public boolean ok() {
            return missingLabels == 0 && malformedLines == 0 && outOfRange == 0 && unknownClass == 0;
        }
    }

    private final int numClasses;

    public DatasetValidator(int numClasses) {
        this.numClasses = numClasses;
    }

    public Report validateSplit(Path imagesDir, Path labelsDir) throws IOException {
        Report r = new Report();
        if (!Files.isDirectory(imagesDir)) {
            r.errors.add("Missing images dir: " + imagesDir);
            return r;
        }
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(imagesDir, "*.{jpg,jpeg,png}")) {
            for (Path img : ds) {
                r.images++;
                String base = stripExt(img.getFileName().toString());
                Path lbl = labelsDir.resolve(base + ".txt");
                if (!Files.exists(lbl)) {
                    r.missingLabels++;
                    continue;
                }
                r.matchedLabels++;
                validateLabelFile(lbl, r);
            }
        }
        LOG.info("Split {} - images={} matched={} missing={} malformed={} oor={} unknown={}",
                imagesDir, r.images, r.matchedLabels, r.missingLabels, r.malformedLines, r.outOfRange, r.unknownClass);
        return r;
    }

    private void validateLabelFile(Path lbl, Report r) throws IOException {
        List<String> lines = Files.readAllLines(lbl);
        for (String line : lines) {
            String s = line.trim();
            if (s.isEmpty()) continue;
            String[] parts = s.split("\\s+");
            if (parts.length != 5) {
                r.malformedLines++;
                r.errors.add("Malformed line in " + lbl + ": " + s);
                continue;
            }
            try {
                int cls = Integer.parseInt(parts[0]);
                if (cls < 0 || cls >= numClasses) {
                    r.unknownClass++;
                    continue;
                }
                for (int i = 1; i < 5; i++) {
                    double v = Double.parseDouble(parts[i]);
                    if (v < 0d || v > 1d) {
                        r.outOfRange++;
                        break;
                    }
                }
            } catch (NumberFormatException e) {
                r.malformedLines++;
                r.errors.add("Number parse error " + lbl + ": " + s);
            }
        }
    }

    private static String stripExt(String name) {
        int dot = name.lastIndexOf('.');
        return dot < 0 ? name : name.substring(0, dot);
    }
}
