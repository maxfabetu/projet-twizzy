package fr.ensem.vision.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class AppConfig {

    private final Map<String, Object> root;

    private AppConfig(Map<String, Object> root) {
        this.root = root;
    }

    @SuppressWarnings("unchecked")
    public static AppConfig load() {
        try (InputStream in = AppConfig.class.getClassLoader().getResourceAsStream("application.yaml")) {
            if (in == null) throw new IllegalStateException("application.yaml not found on classpath");
            ObjectMapper m = new ObjectMapper(new YAMLFactory());
            Map<String, Object> tree = m.readValue(in, Map.class);
            return new AppConfig((Map<String, Object>) tree.get("app"));
        } catch (IOException e) {
            throw new RuntimeException("Failed to load application.yaml", e);
        }
    }

    @SuppressWarnings("unchecked")
    public static AppConfig loadFromFile(Path p) {
        try {
            ObjectMapper m = new ObjectMapper(new YAMLFactory());
            Map<String, Object> tree = m.readValue(Files.newBufferedReader(p), Map.class);
            return new AppConfig((Map<String, Object>) tree.get("app"));
        } catch (IOException e) {
            throw new RuntimeException("Failed to load config: " + p, e);
        }
    }

    @SuppressWarnings("unchecked")
    public <T> T get(String dotted, T defaultValue) {
        String[] parts = dotted.split("\\.");
        Object cur = root;
        for (String p : parts) {
            if (!(cur instanceof Map)) return defaultValue;
            cur = ((Map<String, Object>) cur).get(p);
            if (cur == null) return defaultValue;
        }
        return (T) cur;
    }

    public String getString(String dotted, String def) {
        Object v = get(dotted, (Object) def);
        return v == null ? def : v.toString();
    }

    public int getInt(String dotted, int def) {
        Object v = get(dotted, (Object) def);
        if (v == null) return def;
        return v instanceof Number ? ((Number) v).intValue() : Integer.parseInt(v.toString());
    }

    public double getDouble(String dotted, double def) {
        Object v = get(dotted, (Object) def);
        if (v == null) return def;
        return v instanceof Number ? ((Number) v).doubleValue() : Double.parseDouble(v.toString());
    }

    public boolean getBoolean(String dotted, boolean def) {
        Object v = get(dotted, (Object) def);
        if (v == null) return def;
        return v instanceof Boolean ? (Boolean) v : Boolean.parseBoolean(v.toString());
    }

    @SuppressWarnings("unchecked")
    public List<String> getStringList(String dotted) {
        Object v = get(dotted, null);
        if (v == null) return new ArrayList<>();
        List<Object> raw = (List<Object>) v;
        List<String> out = new ArrayList<>(raw.size());
        for (Object o : raw) out.add(o.toString());
        return out;
    }

    @SuppressWarnings("unchecked")
    public ClassMapping classMapping() {
        Object v = get("classes", null);
        if (!(v instanceof List)) return ClassMapping.defaultMapping();
        List<Map<String, Object>> raw = (List<Map<String, Object>>) v;
        List<ClassMapping.Entry> entries = new ArrayList<>(raw.size());
        for (Map<String, Object> m : raw) {
            int id = ((Number) m.get("id")).intValue();
            String csv = (String) m.get("csv");
            String shortName = (String) m.get("short");
            int yoloId = m.containsKey("yoloId") ? ((Number) m.get("yoloId")).intValue() : id;
            entries.add(new ClassMapping.Entry(id, csv, shortName, yoloId));
        }
        int yoloNumClasses = getInt("yolo.numClasses", entries.size());
        return new ClassMapping(entries, yoloNumClasses);
    }

    public Map<String, Object> asMap() {
        return new LinkedHashMap<>(root);
    }
}
