package fr.ensem.vision.config;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class ClassMapping {

    public static final class Entry {
        private final int id;
        private final String csvName;
        private final String shortName;
        private final int yoloId;

        public Entry(int id, String csvName, String shortName) {
            this(id, csvName, shortName, id);
        }

        public Entry(int id, String csvName, String shortName, int yoloId) {
            this.id = id;
            this.csvName = csvName;
            this.shortName = shortName;
            this.yoloId = yoloId;
        }

        public int getId() { return id; }
        public String getCsvName() { return csvName; }
        public String getShortName() { return shortName; }
        public int getYoloId() { return yoloId; }
    }

    private final List<Entry> entries;
    private final Map<Integer, Entry> byId;
    private final Map<String, Entry> byCsv;
    private final Map<String, Entry> byShort;
    private final Map<Integer, Entry> byYoloId;
    private final int yoloNumClasses;

    public ClassMapping(List<Entry> entries) {
        this(entries, entries.size());
    }

    public ClassMapping(List<Entry> entries, int yoloNumClasses) {
        this.entries = Collections.unmodifiableList(new ArrayList<>(entries));
        this.byId = new HashMap<>();
        this.byCsv = new HashMap<>();
        this.byShort = new HashMap<>();
        this.byYoloId = new HashMap<>();
        for (Entry e : entries) {
            byId.put(e.getId(), e);
            byCsv.put(e.getCsvName(), e);
            byShort.put(e.getShortName(), e);
            byYoloId.put(e.getYoloId(), e);
        }
        this.yoloNumClasses = yoloNumClasses;
    }

    public static ClassMapping defaultMapping() {
        List<Entry> e = new ArrayList<>();
        e.add(new Entry(0, "Speed_limit_50_km_h", "50", 0));
        e.add(new Entry(1, "Speed_limit_70_km_h", "70", 1));
        e.add(new Entry(2, "Speed_limit_90_km_h", "90", 2));
        e.add(new Entry(3, "Speed_limit_110_km_h", "110", 3));
        return new ClassMapping(e, 4);
    }

    public List<Entry> getEntries() { return entries; }
    public Entry byId(int id) { return byId.get(id); }
    public Entry byCsv(String csv) { return byCsv.get(csv); }
    public Entry byShort(String s) { return byShort.get(s); }
    public Entry byYoloId(int yid) { return byYoloId.get(yid); }
    public int size() { return entries.size(); }
    public int yoloNumClasses() { return yoloNumClasses; }

    public List<String> shortNamesInOrder() {
        List<String> n = new ArrayList<>(entries.size());
        for (Entry e : entries) n.add(e.getShortName());
        return n;
    }

    public String labelForId(int id) {
        Entry e = byId.get(id);
        return e == null ? ("class_" + id) : e.getShortName();
    }
}
