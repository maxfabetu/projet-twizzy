package fr.ensem.vision.data;

import com.fasterxml.jackson.databind.MappingIterator;
import com.fasterxml.jackson.dataformat.csv.CsvMapper;
import com.fasterxml.jackson.dataformat.csv.CsvSchema;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class RoboflowCsvReader {

    public static final class Row {
        public final String filename;
        public final Set<String> positiveClasses;

        public Row(String filename, Set<String> positiveClasses) {
            this.filename = filename;
            this.positiveClasses = positiveClasses;
        }
    }

    private final Path csvPath;
    private List<String> headers;

    public RoboflowCsvReader(Path csvPath) {
        this.csvPath = csvPath;
    }

    public List<String> headers() throws IOException {
        if (headers == null) loadHeaders();
        return headers;
    }

    private void loadHeaders() throws IOException {
        try (var lines = Files.lines(csvPath)) {
            String first = lines.findFirst().orElseThrow(() -> new IOException("Empty CSV: " + csvPath));
            String[] parts = first.split(",");
            List<String> hs = new ArrayList<>(parts.length);
            for (String p : parts) hs.add(p.trim());
            this.headers = hs;
        }
    }

    public List<Row> readAll() throws IOException {
        if (headers == null) loadHeaders();
        CsvMapper mapper = new CsvMapper();
        CsvSchema schema = CsvSchema.emptySchema().withHeader();
        List<Row> out = new ArrayList<>();
        try (MappingIterator<Map<String, String>> it = mapper.readerFor(Map.class).with(schema).readValues(csvPath.toFile())) {
            while (it.hasNext()) {
                Map<String, String> row = it.next();
                String filename = firstNonNull(row, "filename", " filename");
                if (filename == null || filename.isBlank()) continue;
                Set<String> positives = new LinkedHashSet<>();
                for (Map.Entry<String, String> e : row.entrySet()) {
                    String k = e.getKey() == null ? "" : e.getKey().trim();
                    if (k.isEmpty() || k.equalsIgnoreCase("filename")) continue;
                    String v = e.getValue() == null ? "" : e.getValue().trim();
                    if ("1".equals(v) || "1.0".equals(v)) positives.add(k);
                }
                out.add(new Row(filename.trim(), positives));
            }
        }
        return out;
    }

    public Map<String, Set<String>> indexByFilename() throws IOException {
        Map<String, Set<String>> idx = new LinkedHashMap<>();
        for (Row r : readAll()) idx.put(r.filename, r.positiveClasses);
        return idx;
    }

    private static String firstNonNull(Map<String, String> row, String... keys) {
        for (String k : keys) {
            String v = row.get(k);
            if (v != null) return v;
        }
        return null;
    }
}
