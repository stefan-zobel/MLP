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
import java.util.SplittableRandom;
import java.util.stream.IntStream;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

/**
 * The training images with a distortion drawn anew for every image on every pass, as an
 * alternative to the stored augmented sets.
 */
public final class AugmentedSet implements PassSource {

    // The mix of the seven stored sets, as closely as the families allow: one part the
    // original, three parts affine and three parts elastic. The one pixel shifted sets get
    // no share of their own because the affine family, which shifts by up to two pixels,
    // already contains them.
    private static final int IDENTITY_WEIGHT = 1;
    private static final int AFFINE_WEIGHT = 3;
    private static final int ELASTIC_WEIGHT = 3;

    private static final float MAX_BYTE = 255.0f;

    private final byte[] source;
    private final MatrixF sourceLabels;
    private final int count;
    private final int w;
    private final int h;

    private final int identityWeight;
    private final int affineWeight;
    private final int totalWeight;

    private final MatrixF images;
    private final MatrixF labels;

    /**
     * Reads the plain MNIST training images and labels; the distortions are drawn later, one
     * per image and pass.
     *
     * @return the MNIST training set, ready to be distorted
     */
    public static AugmentedSet forMnistTraining() {
        try {
            MNISTAugmenter.Images src = MNISTAugmenter.read(MNISTAugmenter.TRAIN_IMAGES);
            return new AugmentedSet(src.data, src.count, src.w, src.h, MNIST.getTrainingSetLabels());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Reads the plain EMNIST training images and labels, already turned upright.
     *
     * @return the EMNIST training set, ready to be distorted
     */
    public static AugmentedSet forEmnistTraining() {
        return forEmnistTraining(IDENTITY_WEIGHT, AFFINE_WEIGHT, ELASTIC_WEIGHT);
    }

    /**
     * The EMNIST training set under a chosen mix of distortions; weights of {@code 1, 0, 0}
     * leave every image alone and give an unaugmented set.
     *
     * @param identityWeight share of images that stay as they are
     * @param affineWeight share of images under a random rotation, scaling and translation
     * @param elasticWeight share of images under a smoothed random displacement field
     * @return the EMNIST training set, ready to be distorted
     */
    public static AugmentedSet forEmnistTraining(int identityWeight, int affineWeight, int elasticWeight) {
        try {
            MNISTAugmenter.Images src = EMNIST.readTrainingImages();
            return new AugmentedSet(src.data, src.count, src.w, src.h, EMNIST.getTrainingSetLabels(), identityWeight,
                    affineWeight, elasticWeight);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    AugmentedSet(byte[] source, int count, int w, int h, MatrixF labels) {
        this(source, count, w, h, labels, IDENTITY_WEIGHT, AFFINE_WEIGHT, ELASTIC_WEIGHT);
    }

    AugmentedSet(byte[] source, int count, int w, int h, MatrixF labels, int identityWeight, int affineWeight,
            int elasticWeight) {
        if (source.length != count * w * h) {
            throw new IllegalArgumentException(
                    "source.length " + source.length + " != " + count + " * " + w + " * " + h);
        }
        if (labels.numColumns() != count) {
            throw new IllegalArgumentException("got " + labels.numColumns() + " labels for " + count + " images");
        }
        if (identityWeight < 0 || affineWeight < 0 || elasticWeight < 0) {
            throw new IllegalArgumentException(
                    "negative weight: " + identityWeight + ", " + affineWeight + ", " + elasticWeight);
        }
        if (identityWeight + affineWeight + elasticWeight <= 0) {
            throw new IllegalArgumentException("all three weights are zero");
        }
        this.source = source;
        this.sourceLabels = labels;
        this.count = count;
        this.w = w;
        this.h = h;
        this.identityWeight = identityWeight;
        this.affineWeight = affineWeight;
        this.totalWeight = identityWeight + affineWeight + elasticWeight;
        this.images = Matrices.createF(w * h, count);
        this.labels = Matrices.createF(labels.numRows(), count);
    }

    /**
     * The distorted images of the latest pass, one per column, scaled into {@code [0, 1]}.
     * The matrix is refilled in place and holds zeros until the first pass.
     *
     * @return the {@code w * h x count} image matrix
     */
    @Override
    public MatrixF images() {
        return images;
    }

    /**
     * The labels belonging to {@link #images}, column by column.
     *
     * @return the {@code classes x count} label matrix
     */
    @Override
    public MatrixF labels() {
        return labels;
    }

    /**
     * Refills {@link #images} and {@link #labels} with a fresh pass: every image appears
     * exactly once, in a new order and under a newly drawn distortion.
     *
     * @param rnd source of the order and of the per-image draws
     */
    @Override
    public void regenerate(SplittableRandom rnd) {
        // The order and the per-image seeds are drawn here, before any thread starts, so
        // that the pass depends on rnd alone and not on how the work happens to be split.
        int[] order = permutation(rnd);
        long[] seeds = new long[count];
        for (int j = 0; j < count; ++j) {
            seeds[j] = rnd.nextLong();
        }

        // Both matrices are column-major with contiguous columns, so image j owns the
        // pixels slots from j * pixels and its label the labelRows slots from j * labelRows.
        final int pixels = w * h;
        final int labelRows = sourceLabels.numRows();
        final float[] out = images.getArrayUnsafe();
        final float[] dstLabels = labels.getArrayUnsafe();
        final float[] srcLabels = sourceLabels.getArrayUnsafe();

        IntStream.range(0, count).parallel().forEach(j -> {
            byte[] image = new byte[pixels];
            System.arraycopy(source, order[j] * pixels, image, 0, pixels);
            byte[] distorted = distort(image, seeds[j]);
            int base = j * pixels;
            for (int p = 0; p < pixels; ++p) {
                // dividing by 255 rather than rescaling by the observed extremes: on MNIST
                // the two agree, but a per-pass rescale would let the scaling depend on
                // what that pass happened to draw
                out[base + p] = (distorted[p] & 0xFF) / MAX_BYTE;
            }
            System.arraycopy(srcLabels, order[j] * labelRows, dstLabels, j * labelRows, labelRows);
        });
    }

    private byte[] distort(byte[] image, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        int draw = rnd.nextInt(totalWeight);
        if (draw < identityWeight) {
            return image;
        }
        if (draw < identityWeight + affineWeight) {
            return MNISTAugmenter.randomAffine(image, w, h, rnd);
        }
        return MNISTAugmenter.randomElastic(image, w, h, rnd);
    }

    private int[] permutation(SplittableRandom rnd) {
        int[] order = new int[count];
        for (int i = 0; i < count; ++i) {
            order[i] = i;
        }
        for (int i = count - 1; i > 0; --i) {
            int j = rnd.nextInt(i + 1);
            int tmp = order[i];
            order[i] = order[j];
            order[j] = tmp;
        }
        return order;
    }
}
