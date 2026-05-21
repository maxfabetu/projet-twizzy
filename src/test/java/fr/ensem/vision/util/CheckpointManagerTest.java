package fr.ensem.vision.util;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CheckpointManagerTest {

    @Test
    void roundTrip(@TempDir Path tmp) throws Exception {
        Path file = tmp.resolve("ck.json");
        CheckpointManager cm = new CheckpointManager(file, "job-1", 5);
        cm.loadOrCreate("stage-a", 100L);
        for (int i = 0; i < 17; i++) cm.update(1L);
        cm.close();
        assertTrue(Files.exists(file));

        CheckpointManager cm2 = new CheckpointManager(file, "job-1", 5);
        CheckpointManager.State s = cm2.loadOrCreate("stage-a", 100L);
        assertEquals(17L, s.processed);
        assertEquals("stage-a", s.stage);
        assertEquals("job-1", s.jobId);
    }

    @Test
    void jobMismatchRestarts(@TempDir Path tmp) throws Exception {
        Path file = tmp.resolve("ck.json");
        CheckpointManager cm = new CheckpointManager(file, "job-1", 1);
        cm.loadOrCreate("stage-a", 50L);
        cm.update(10L);
        cm.close();

        CheckpointManager cm2 = new CheckpointManager(file, "job-2", 1);
        CheckpointManager.State s = cm2.loadOrCreate("stage-a", 50L);
        assertEquals(0L, s.processed);
    }
}
