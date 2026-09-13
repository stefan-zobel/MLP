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

import java.util.Arrays;
import java.util.SplittableRandom;
import java.util.stream.IntStream;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

/**
 * A fixed number of passes drawn once and then handed out again and again, so that a run can
 * be compared against the same distortions kept fresh instead of frozen.
 */
public final class FrozenAugmentedSet implements PassSource {

    private static final float MAX_BYTE = 255.0f;

    // the frozen passes, one entry each: the distorted pixels and the class of every column
    private final byte[][] frozenPixels;
    private final byte[][] frozenLabels;

    private final int count;
    private final int pixels;
    private final int classes;

    private final MatrixF images;
    private final MatrixF labels;

    private int served = 0;

    /**
     * Draws the given number of passes from {@code live} and keeps them.
     *
     * @param live the generator to freeze
     * @param passes how many passes to keep
     * @param rnd source of the draws, consumed once here and never again
     * @return the frozen passes
     */
    public static FrozenAugmentedSet of(AugmentedSet live, int passes, SplittableRandom rnd) {
        if (passes < 1) {
            throw new IllegalArgumentException("passes: " + passes);
        }
        return new FrozenAugmentedSet(live, passes, rnd);
    }

    private FrozenAugmentedSet(AugmentedSet live, int passes, SplittableRandom rnd) {
        this.count = live.images().numColumns();
        this.pixels = live.images().numRows();
        this.classes = live.labels().numRows();
        this.frozenPixels = new byte[passes][];
        this.frozenLabels = new byte[passes][];
        // Kept as bytes, not as the floats they arrive in. Seven passes over EMNIST are
        // 590 MB this way and 2.4 GB the other, and the larger figure does not fit beside
        // everything else a run holds. Nothing is lost: the values came from bytes.
        for (int p = 0; p < passes; ++p) {
            live.regenerate(rnd);
            frozenPixels[p] = toBytes(live.images());
            frozenLabels[p] = toClasses(live.labels());
        }
        this.images = Matrices.createF(pixels, count);
        this.labels = Matrices.createF(classes, count);
    }

    /**
     * The images of the pass served last, one per column, scaled into {@code [0, 1]}.
     *
     * @return the {@code pixels x count} image matrix
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
     * Serves the next frozen pass, in a newly drawn order.
     *
     * @param rnd source of the order; the images themselves are already fixed
     */
    @Override
    public void regenerate(SplittableRandom rnd) {
        // The passes come round in the order they were drawn. Only the column order is new,
        // which is what the stored sets got from a reshuffle between epochs: the batches
        // differ from one epoch to the next, their contents do not.
        final byte[] srcPixels = frozenPixels[served % frozenPixels.length];
        final byte[] srcLabels = frozenLabels[served % frozenLabels.length];
        ++served;

        int[] order = permutation(rnd);
        final float[] out = images.getArrayUnsafe();
        final float[] dstLabels = labels.getArrayUnsafe();
        Arrays.fill(dstLabels, 0.0f);

        IntStream.range(0, count).parallel().forEach(j -> {
            int from = order[j] * pixels;
            int base = j * pixels;
            for (int p = 0; p < pixels; ++p) {
                out[base + p] = (srcPixels[from + p] & 0xFF) / MAX_BYTE;
            }
            dstLabels[j * classes + (srcLabels[order[j]] & 0xFF)] = 1.0f;
        });
    }

    private static byte[] toBytes(MatrixF images) {
        float[] in = images.getArrayUnsafe();
        byte[] out = new byte[in.length];
        for (int i = 0; i < in.length; ++i) {
            out[i] = (byte) Math.round(in[i] * MAX_BYTE);
        }
        return out;
    }

    // the one hot column of every image, as the class it stands for
    private static byte[] toClasses(MatrixF labels) {
        final int classes = labels.numRows();
        final int count = labels.numColumns();
        float[] in = labels.getArrayUnsafe();
        byte[] out = new byte[count];
        for (int j = 0; j < count; ++j) {
            int base = j * classes;
            int hot = -1;
            for (int c = 0; c < classes; ++c) {
                if (in[base + c] != 0.0f) {
                    hot = c;
                    break;
                }
            }
            if (hot < 0) {
                throw new IllegalArgumentException("column " + j + " is not one hot");
            }
            out[j] = (byte) hot;
        }
        return out;
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
