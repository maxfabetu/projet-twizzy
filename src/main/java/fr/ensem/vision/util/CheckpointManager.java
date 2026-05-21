package fr.ensem.vision.util;

import com.fasterxml.jackson.annotation.JsonInclude;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;

public final class CheckpointManager {

    @JsonInclude(JsonInclude.Include.NON_NULL)
    public static final class State {
        public String jobId;
        public String stage;
        public long processed;
        public long total;
        public Instant updatedAt;
        public Map<String, Object> custom = new LinkedHashMap<>();

        public State() {}

        public State(String jobId, String stage, long processed, long total) {
            this.jobId = jobId;
            this.stage = stage;
            this.processed = processed;
            this.total = total;
            this.updatedAt = Instant.now();
        }
    }

    private final Path file;
    private final int saveEveryN;
    private final String jobId;
    private State state;
    private long sinceLastSave = 0;

    public CheckpointManager(Path file, String jobId, int saveEveryN) {
        this.file = file;
        this.jobId = jobId;
        this.saveEveryN = Math.max(1, saveEveryN);
    }

    public State loadOrCreate(String stage, long total) throws IOException {
        if (Files.exists(file)) {
            try {
                State loaded = AtomicJsonWriter.read(file, State.class);
                if (loaded != null && jobId.equals(loaded.jobId) && stage.equals(loaded.stage)) {
                    this.state = loaded;
                    return state;
                }
            } catch (IOException ignored) {}
        }
        this.state = new State(jobId, stage, 0L, total);
        save();
        return state;
    }

    public void update(long processedIncrement) throws IOException {
        if (state == null) throw new IllegalStateException("Checkpoint not initialised");
        state.processed += processedIncrement;
        state.updatedAt = Instant.now();
        sinceLastSave += processedIncrement;
        if (sinceLastSave >= saveEveryN) {
            save();
            sinceLastSave = 0;
        }
    }

    public void putCustom(String key, Object value) {
        if (state == null) return;
        state.custom.put(key, value);
    }

    public void save() throws IOException {
        if (state == null) return;
        state.updatedAt = Instant.now();
        AtomicJsonWriter.write(file, state);
    }

    public void close() throws IOException {
        save();
    }

    public State state() { return state; }
}
