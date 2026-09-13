/*
 * Copyright 2024, 2026 Stefan Zobel
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

import net.jamu.matrix.MatrixF;

/**
 * A loader for Yann LeCuns MNIST datasets.
 */
public final class MNIST {

    private static final int NUMBER_OF_DISTINCT_LABELS = 10;

    private static final String TRAIN_IMAGES = "./data/mnist/train-images.idx3-ubyte";
    private static final String TRAIN_IMAGES_LEFT = "./data/mnist/train-images-left.idx3-ubyte";
    private static final String TRAIN_IMAGES_RIGHT = "./data/mnist/train-images-right.idx3-ubyte";
    private static final String TRAIN_IMAGES_AFFINE1 = "./data/mnist/train-images-affine1.idx3-ubyte";
    private static final String TRAIN_IMAGES_AFFINE2 = "./data/mnist/train-images-affine2.idx3-ubyte";
    private static final String TRAIN_IMAGES_ELASTIC1 = "./data/mnist/train-images-elastic1.idx3-ubyte";
    private static final String TRAIN_IMAGES_ELASTIC2 = "./data/mnist/train-images-elastic2.idx3-ubyte";
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
     * Loads the first affinely distorted copy of the training images into a
     * {@code 784 x 60_000} matrix.
     *
     * @return MNIST training set images under a random rotation, scaling and translation
     */
    public static MatrixF getTrainingSetImagesAffine1() {
        try {
            return readImages(TRAIN_IMAGES_AFFINE1);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Loads the second affinely distorted copy of the training images into a
     * {@code 784 x 60_000} matrix.
     *
     * @return MNIST training set images under a random rotation, scaling and translation
     */
    public static MatrixF getTrainingSetImagesAffine2() {
        try {
            return readImages(TRAIN_IMAGES_AFFINE2);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Loads the first elastically distorted copy of the training images into a
     * {@code 784 x 60_000} matrix.
     *
     * @return MNIST training set images under a smoothed random displacement field
     */
    public static MatrixF getTrainingSetImagesElastic1() {
        try {
            return readImages(TRAIN_IMAGES_ELASTIC1);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Loads the second elastically distorted copy of the training images into a
     * {@code 784 x 60_000} matrix.
     *
     * @return MNIST training set images under a smoothed random displacement field
     */
    public static MatrixF getTrainingSetImagesElastic2() {
        try {
            return readImages(TRAIN_IMAGES_ELASTIC2);
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
        return Idx.readImages(path);
    }

    private static MatrixF readLabels(String path) throws IOException {
        return Idx.readLabels(path, NUMBER_OF_DISTINCT_LABELS);
    }

    private MNIST() {
        throw new AssertionError();
    }
}
