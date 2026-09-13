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
package math.ml.mlp;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

/**
 * Turns the {@code channels x (batch * height * width)} layout of {@link Conv2D} into the
 * {@code features x batch} layout a {@link Hidden} expects.
 *
 * <p>This is a per-sample transpose of the channel against the position axis and not a free
 * reinterpretation, except for a single channel, where it degenerates to a copy.
 */
public class Flatten extends AbstractLayer {

    private final int channels;
    private final int h;
    private final int w;

    private MatrixF output;
    private MatrixF inputGrads;
    /** The batch size the buffers above were sized for, or -1 before the first pass. */
    private int batch = -1;

    /**
     * Creates a flattening layer for a known geometry.
     *
     * @param channels channels of the input
     * @param h        height of the input
     * @param w        width of the input
     */
    public Flatten(int channels, int h, int w) {
        if (channels <= 0 || h <= 0 || w <= 0) {
            throw new IllegalArgumentException(
                    "channels " + channels + ", height " + h + " and width " + w + " must all be positive");
        }
        this.channels = channels;
        this.h = h;
        this.w = w;
    }

    /** The number of features one sample turns into. */
    public int features() {
        return channels * h * w;
    }

    /**
     * Writes the conv layout {@code src} into the {@code features x batch} layout
     * {@code dst}. Package-private because {@link Unflatten} runs the same mapping the
     * other way round, and one copy of the index arithmetic is one place to get it wrong.
     *
     * @param src      the conv layout, channels x (m * spatial)
     * @param dst      the flat layout, (channels * spatial) x m
     * @param channels channels of the conv layout
     * @param spatial  positions per sample
     * @param m        number of samples
     */
    static void gather(float[] src, float[] dst, int channels, int spatial, int m) {
        int features = channels * spatial;
        for (int s = 0; s < m; ++s) {
            int sample = s * spatial * channels;
            int target = s * features;
            for (int c = 0; c < channels; ++c) {
                for (int p = 0; p < spatial; ++p) {
                    dst[target + c * spatial + p] = src[sample + p * channels + c];
                }
            }
        }
    }

    /**
     * The inverse of {@link #gather}: writes the flat layout {@code src} into the conv
     * layout {@code dst}.
     *
     * @param src      the flat layout, (channels * spatial) x m
     * @param dst      the conv layout, channels x (m * spatial)
     * @param channels channels of the conv layout
     * @param spatial  positions per sample
     * @param m        number of samples
     */
    static void scatter(float[] src, float[] dst, int channels, int spatial, int m) {
        int features = channels * spatial;
        for (int s = 0; s < m; ++s) {
            int sample = s * spatial * channels;
            int source = s * features;
            for (int c = 0; c < channels; ++c) {
                for (int p = 0; p < spatial; ++p) {
                    dst[sample + p * channels + c] = src[source + c * spatial + p];
                }
            }
        }
    }

    @Override
    public MatrixF forward(MatrixF in) {
        int m = batchOf(in);
        ensureBuffers(m);
        gather(in.getArrayUnsafe(), output.getArrayUnsafe(), channels, h * w, m);
        return output;
    }

    @Override
    public MatrixF backward(MatrixF outputGrads) {
        if (mode == NetworkMode.INFER) {
            return null;
        }
        int m = outputGrads.numColumns();
        ensureBuffers(m);
        scatter(outputGrads.getArrayUnsafe(), inputGrads.getArrayUnsafe(), channels, h * w, m);
        return inputGrads;
    }

    private int batchOf(MatrixF in) {
        if (in.numRows() != channels) {
            throw new IllegalArgumentException("expected " + channels + " channels, got " + in.numRows());
        }
        int spatial = h * w;
        if (in.numColumns() % spatial != 0) {
            throw new IllegalArgumentException(
                    in.numColumns() + " columns is not a whole number of " + h + "x" + w + " images");
        }
        return in.numColumns() / spatial;
    }

    private void ensureBuffers(int m) {
        if (batch == m) {
            return;
        }
        output = Matrices.createF(channels * h * w, m);
        inputGrads = Matrices.createF(channels, m * h * w);
        batch = m;
    }
}
