/*
 * Copyright 2026 Stefan Zobel
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package math.ml.loader;

import java.io.IOException;
import java.io.UncheckedIOException;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

/**
 * A loader for the balanced split of the EMNIST handwritten character dataset.
 */
public final class EMNIST {

    // 10 digits, the 26 capitals, and the 11 lower case letters whose shape differs from
    // their capital: a b d e f g h n q r t
    private static final int NUMBER_OF_DISTINCT_LABELS = 47;

    static final String TRAIN_IMAGES = "./data/emnist/train-images.idx3-ubyte";
    private static final String TRAIN_LABELS = "./data/emnist/train-labels.idx1-ubyte";
    private static final String TEST_IMAGES = "./data/emnist/test-images.idx3-ubyte";
    private static final String TEST_LABELS = "./data/emnist/test-labels.idx1-ubyte";

    /**
     * Loads the training images into a {@code 784 x 112_800} matrix.
     *
     * @return EMNIST training set images
     */
    public static MatrixF getTrainingSetImages() {
        return readImages(TRAIN_IMAGES);
    }

    /**
     * Loads the test images into a {@code 784 x 18_800} matrix.
     *
     * @return EMNIST test set images
     */
    public static MatrixF getTestSetImages() {
        return readImages(TEST_IMAGES);
    }

    /**
     * Loads the training labels into a {@code 47 x 112_800} matrix.
     *
     * @return EMNIST training set labels
     */
    public static MatrixF getTrainingSetLabels() {
        return readLabels(TRAIN_LABELS);
    }

    /**
     * Loads the test labels into a {@code 47 x 18_800} matrix.
     *
     * @return EMNIST test set labels
     */
    public static MatrixF getTestSetLabels() {
        return readLabels(TEST_LABELS);
    }

    // package-private because AugmentedSet distorts the raw bytes rather than a matrix
    static MNISTAugmenter.Images readTrainingImages() throws IOException {
        return readTransposed(TRAIN_IMAGES);
    }

    // EMNIST stores every image with its rows and columns interchanged, so a character read
    // the way MNIST is read comes out lying on its side and mirrored. Everything downstream
    // -- the augmenter, the augmented set, the test matrix -- is spared knowing that because
    // the images are put upright here, at the one place they enter the program.
    private static MNISTAugmenter.Images readTransposed(String path) throws IOException {
        MNISTAugmenter.Images src = MNISTAugmenter.read(path);
        final int pixels = src.w * src.h;
        byte[] out = new byte[src.data.length];
        byte[] image = new byte[pixels];
        for (int i = 0; i < src.count; ++i) {
            System.arraycopy(src.data, i * pixels, image, 0, pixels);
            System.arraycopy(transpose(image, src.w, src.h), 0, out, i * pixels, pixels);
        }
        // width and height trade places under a transpose; they are equal for EMNIST, but
        // saying so here keeps the holder honest
        return new MNISTAugmenter.Images(src.count, src.h, src.w, out);
    }

    // package-private so that a test can name it
    static byte[] transpose(byte[] image, int w, int h) {
        if (image.length != w * h) {
            throw new IllegalArgumentException("image.length " + image.length + " != " + w + " * " + h);
        }
        byte[] out = new byte[image.length];
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                // the result is h wide and w tall
                out[x * h + y] = image[y * w + x];
            }
        }
        return out;
    }

    private static MatrixF readImages(String path) {
        try {
            MNISTAugmenter.Images images = readTransposed(path);
            // both layouts run image by image, and within an image row by row, so the bytes
            // land in the matrix one for one
            MatrixF matrix = Matrices.createF(images.w * images.h, images.count);
            float[] out = matrix.getArrayUnsafe();
            for (int i = 0; i < out.length; ++i) {
                out[i] = images.data[i] & 0xFF;
            }
            return matrix;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static MatrixF readLabels(String path) {
        try {
            return Idx.readLabels(path, NUMBER_OF_DISTINCT_LABELS);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private EMNIST() {
        throw new AssertionError();
    }
}
