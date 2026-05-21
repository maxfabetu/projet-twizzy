package fr.ensem.vision.eval;

public final class PrecisionRecallF1 {

    public final double precision;
    public final double recall;
    public final double f1;
    public final int tp;
    public final int fp;
    public final int fn;

    public PrecisionRecallF1(int tp, int fp, int fn) {
        this.tp = tp;
        this.fp = fp;
        this.fn = fn;
        double p = (tp + fp) == 0 ? 0d : tp / (double) (tp + fp);
        double r = (tp + fn) == 0 ? 0d : tp / (double) (tp + fn);
        this.precision = p;
        this.recall = r;
        this.f1 = (p + r) == 0d ? 0d : 2d * p * r / (p + r);
    }

    @Override
    public String toString() {
        return String.format("P=%.3f R=%.3f F1=%.3f (tp=%d fp=%d fn=%d)", precision, recall, f1, tp, fp, fn);
    }
}
