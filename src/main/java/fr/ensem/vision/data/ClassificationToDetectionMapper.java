package fr.ensem.vision.data;

import fr.ensem.vision.config.ClassMapping;

import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

public final class ClassificationToDetectionMapper {

    private final ClassMapping mapping;

    public ClassificationToDetectionMapper(ClassMapping mapping) {
        this.mapping = mapping;
    }

    public Map<String, Set<Integer>> mapPositives(Map<String, Set<String>> csvIndex) {
        Map<String, Set<Integer>> out = new LinkedHashMap<>();
        for (Map.Entry<String, Set<String>> e : csvIndex.entrySet()) {
            Set<Integer> ids = new HashSet<>();
            for (String csv : e.getValue()) {
                ClassMapping.Entry m = mapping.byCsv(csv);
                if (m != null) ids.add(m.getId());
            }
            if (!ids.isEmpty()) out.put(e.getKey(), ids);
        }
        return out;
    }

    public boolean isOneOfTargetClasses(Set<String> csvClasses) {
        for (String c : csvClasses) {
            if (mapping.byCsv(c) != null) return true;
        }
        return false;
    }
}
