/*
 * Copyright 2024 Stefan Zobel
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

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.UncheckedIOException;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

/**
 * A loader for Yann LeCuns MNIST datasets.
 */
public final class MNIST {

    private static final int NUMBER_OF_DISTINCT_LABELS = 10;

    private static final MatrixF[] ONE_HOT = new MatrixF[NUMBER_OF_DISTINCT_LABELS];

    private static final String TRAIN_IMAGES = "./data/mnist/train-images.idx3-ubyte";
    private static final String TRAIN_IMAGES_LEFT = "./data/mnist/train-images-left.idx3-ubyte";
    private static final String TRAIN_IMAGES_RIGHT = "./data/mnist/train-images-right.idx3-ubyte";
    private static final String TRAIN_LABELS = "./data/mnist/train-labels.idx1-ubyte";
    private static final String TEST_IMAGES = "./data/mnist/t10k-images.idx3-ubyte";
    private static final String TEST_LABELS = "./data/mnist/t10k-labels.idx1-ubyte";

    /**
     * Loads the training images into a {@code 784 x 60_000} matrix.
     * 
     * @return MNIST training set images
     */
    public static MatrixF getTrainingSetImages() {
        try {
            return readImages(TRAIN_IMAGES);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Loads the training images (shifted one pixel to the left) into a
     * {@code 784 x 60_000} matrix.
     * 
     * @return MNIST training set images shifted one pixel to the left
     */
    public static MatrixF getTrainingSetImagesLeft() {
        try {
            return readImages(TRAIN_IMAGES_LEFT);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Loads the training images (shifted one pixel to the right) into a
     * {@code 784 x 60_000} matrix.
     * 
     * @return MNIST training set images shifted one pixel to the right
     */
    public static MatrixF getTrainingSetImagesRight() {
        try {
            return readImages(TRAIN_IMAGES_RIGHT);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Loads the test images into a {@code 784 x 10_000} matrix.
     * 
     * @return MNIST test set images
     */
    public static MatrixF getTestSetImages() {
        try {
            return readImages(TEST_IMAGES);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Loads the training labels into a {@code 10 x 60_000} matrix.
     * 
     * @return MNIST training set labels
     */
    public static MatrixF getTrainingSetLabels() {
        try {
            return readLabels(TRAIN_LABELS);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Loads the test labels into a {@code 10 x 10_000} matrix.
     * 
     * @return MNIST test set labels
     */
    public static MatrixF getTestSetLabels() {
        try {
            return readLabels(TEST_LABELS);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static MatrixF readImages(String path) throws IOException {
        try (DataInputStream ds = getDataInputStream(path)) {
            int imageCount = ds.readInt();
            int rowPixelCount = ds.readInt();
            int colPixelCount = ds.readInt();
            final int matrixRowCount = rowPixelCount * colPixelCount;
            // we store each image in a column of the returned matrix
            MatrixF images = Matrices.createF(matrixRowCount, imageCount);
            // images in the LeCun files are stored in row-major, so we store them line by
            // line into our column
            for (int col = 0; col < imageCount; ++col) {
                for (int row = 0; row < matrixRowCount; ++row) {
                    images.set(row, col, ds.readUnsignedByte());
                }
            }
            return images;
        }
    }

    private static MatrixF readLabels(String path) throws IOException {
        try (DataInputStream ds = getDataInputStream(path)) {
            int labelCount = ds.readInt();
            MatrixF labels = Matrices.createF(NUMBER_OF_DISTINCT_LABELS, labelCount);
            for (int i = 0; i < labelCount; ++i) {
                int label = ds.readUnsignedByte();
                labels.setColumnInplace(i, ONE_HOT[label]);
            }
            return labels;
        }
    }

    private static DataInputStream getDataInputStream(String path) throws IOException {
        DataInputStream ds = new DataInputStream(new BufferedInputStream(new FileInputStream(path)));
        // throw away magic number
        ds.readInt();
        return ds;
    }

    // initialize ONE_HOT templates
    static {
        for (int i = 0; i < NUMBER_OF_DISTINCT_LABELS; ++i) {
            // one hot column vector template
            ONE_HOT[i] = Matrices.createF(NUMBER_OF_DISTINCT_LABELS, 1);
            // row i is 1, all other rows are 0
            ONE_HOT[i].set(i, 0, 1.0f);
        }
    }

    private MNIST() {
        throw new AssertionError();
    }
}
