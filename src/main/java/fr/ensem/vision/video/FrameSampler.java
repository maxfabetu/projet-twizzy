package fr.ensem.vision.video;

public final class FrameSampler {

    private final int step;
    private long counter = -1;

    public FrameSampler(int step) {
        this.step = Math.max(1, step);
    }

    public boolean accept() {
        counter++;
        return counter % step == 0;
    }

    public int step() { return step; }
}
