package fr.ensem.vision.util;

import org.bytedeco.opencv.opencv_core.Mat;

import java.util.ArrayDeque;
import java.util.Deque;

public final class MatPool implements AutoCloseable {

    private final Deque<Mat> available = new ArrayDeque<>();
    private final int capacity;

    public MatPool(int capacity) {
        this.capacity = Math.max(1, capacity);
    }

    public synchronized Mat acquire() {
        Mat m = available.pollFirst();
        return m == null ? new Mat() : m;
    }

    public synchronized void release(Mat m) {
        if (m == null) return;
        if (available.size() >= capacity) {
            m.release();
            m.close();
            return;
        }
        available.addLast(m);
    }

    @Override
    public synchronized void close() {
        for (Mat m : available) {
            m.release();
            m.close();
        }
        available.clear();
    }
}
