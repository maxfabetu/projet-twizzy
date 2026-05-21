package fr.ensem.vision.data;

import fr.ensem.vision.config.ClassMapping;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public final class DataYamlWriter {

    private DataYamlWriter() {}

    public static void write(Path yamlFile, Path yoloRoot, ClassMapping mapping) throws IOException {
        Path parent = yamlFile.getParent();
        if (parent != null) Files.createDirectories(parent);
        List<String> lines = new ArrayList<>();
        lines.add("path: " + yoloRoot.toAbsolutePath().toString().replace('\\', '/'));
        lines.add("train: train/images");
        lines.add("val: valid/images");
        lines.add("test: test/images");
        lines.add("nc: " + mapping.size());
        StringBuilder names = new StringBuilder("names: [");
        for (int i = 0; i < mapping.size(); i++) {
            if (i > 0) names.append(", ");
            names.append('\'').append(mapping.byId(i).getShortName()).append('\'');
        }
        names.append("]");
        lines.add(names.toString());
        Files.write(yamlFile, lines, StandardCharsets.UTF_8);
    }
}
