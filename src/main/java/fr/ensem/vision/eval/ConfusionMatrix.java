package fr.ensem.vision.eval;

public final class ConfusionMatrix {

    private final int numClasses;
    private final int[][] matrix;

    public ConfusionMatrix(int numClasses) {
        this.numClasses = numClasses + 1;
        this.matrix = new int[this.numClasses][this.numClasses];
    }

    public void increment(int actual, int predicted) {
        int a = actual < 0 ? numClasses - 1 : actual;
        int p = predicted < 0 ? numClasses - 1 : predicted;
        matrix[a][p]++;
    }

    public int get(int actual, int predicted) {
        int a = actual < 0 ? numClasses - 1 : actual;
        int p = predicted < 0 ? numClasses - 1 : predicted;
        return matrix[a][p];
    }

    public int dim() { return numClasses; }
    public int[][] raw() { return matrix; }
}
