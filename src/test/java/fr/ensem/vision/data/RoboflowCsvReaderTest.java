package fr.ensem.vision.data;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RoboflowCsvReaderTest {

    @Test
    void parsesPositiveClasses(@TempDir Path tmp) throws Exception {
        Path csv = tmp.resolve("_classes.csv");
        String content = String.join("\n",
                "filename,Speed_limit_50_km_h,Speed_limit_70_km_h,Other",
                "a.jpg,1,0,0",
                "b.jpg,0,1,1",
                "c.jpg,0,0,0") + "\n";
        Files.writeString(csv, content, StandardCharsets.UTF_8);

        RoboflowCsvReader r = new RoboflowCsvReader(csv);
        List<RoboflowCsvReader.Row> rows = r.readAll();
        assertEquals(3, rows.size());
        Map<String, Set<String>> idx = r.indexByFilename();
        assertTrue(idx.get("a.jpg").contains("Speed_limit_50_km_h"));
        assertTrue(idx.get("b.jpg").contains("Speed_limit_70_km_h"));
        assertTrue(idx.get("b.jpg").contains("Other"));
        assertEquals(0, idx.get("c.jpg").size());
    }
}
