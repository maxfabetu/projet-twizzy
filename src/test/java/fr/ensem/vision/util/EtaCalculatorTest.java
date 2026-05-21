package fr.ensem.vision.util;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EtaCalculatorTest {

    @Test
    void emptyWindowMeansZeroFps() {
        EtaCalculator eta = new EtaCalculator(5);
        assertEquals(0d, eta.fps(), 1e-9);
    }

    @Test
    void firstTickDoesNotProduceMeasurement() throws InterruptedException {
        EtaCalculator eta = new EtaCalculator(3);
        eta.tick();
        assertEquals(0d, eta.averageItemSeconds(), 1e-9);
        Thread.sleep(20);
        eta.tick();
        assertTrue(eta.averageItemSeconds() > 0d);
    }

    @Test
    void slidingWindowKeepsOnlyLastN() throws InterruptedException {
        EtaCalculator eta = new EtaCalculator(2);
        for (int i = 0; i < 5; i++) {
            eta.tick();
            Thread.sleep(5);
        }
        assertTrue(eta.fps() > 0d);
    }

    @Test
    void etaIsZeroWhenProcessedReachesTotal() {
        EtaCalculator eta = new EtaCalculator(5);
        eta.setTotal(2);
        eta.setProcessed(2);
        assertEquals(0L, eta.estimatedRemaining().getSeconds());
    }
}
