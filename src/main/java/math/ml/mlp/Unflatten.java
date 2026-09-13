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
 * Turns the {@code features x batch} layout the loaders produce into the
 * {@code channels x (batch * height * width)} layout {@link Conv2D} expects, which is
 * {@link Flatten} the other way round.
 *
 * <p>For a single channel the two layouts hold their elements in the same order, so this
 * degenerates to a copy; jamu offers no way to wrap an existing array, which is why even
 * that copy has to happen.
 */
public class Unflatten extends AbstractLayer {

    private final int channels;
    private final int h;
    private final int w;

    private MatrixF output;
    private MatrixF inputGrads;
    /** The batch size the buffers above were sized for, or -1 before the first pass. */
    private int batch = -1;

    /**
     * Creates an unflattening layer for a known geometry.
     *
     * @param channels channels of the output
     * @param h        height of the output
     * @param w        width of the output
     */
    public Unflatten(int channels, int h, int w) {
        if (channels <= 0 || h <= 0 || w <= 0) {
            throw new IllegalArgumentException(
                    "channels " + channels + ", height " + h + " and width " + w + " must all be positive");
        }
        this.channels = channels;
        this.h = h;
        this.w = w;
    }

    /** The number of features one sample is expected to arrive as. */
    public int features() {
        return channels * h * w;
    }

    @Override
    public MatrixF forward(MatrixF in) {
        int features = channels * h * w;
        if (in.numRows() != features) {
            throw new IllegalArgumentException("expected " + features + " features, got " + in.numRows());
        }
        int m = in.numColumns();
        ensureBuffers(m);
        Flatten.scatter(in.getArrayUnsafe(), output.getArrayUnsafe(), channels, h * w, m);
        return output;
    }

    @Override
    public MatrixF backward(MatrixF outputGrads) {
        if (mode == NetworkMode.INFER) {
            return null;
        }
        int spatial = h * w;
        if (outputGrads.numColumns() % spatial != 0) {
            throw new IllegalArgumentException(
                    outputGrads.numColumns() + " columns is not a whole number of " + h + "x" + w + " images");
        }
        int m = outputGrads.numColumns() / spatial;
        ensureBuffers(m);
        Flatten.gather(outputGrads.getArrayUnsafe(), inputGrads.getArrayUnsafe(), channels, spatial, m);
        return inputGrads;
    }

    private void ensureBuffers(int m) {
        if (batch == m) {
            return;
        }
        output = Matrices.createF(channels, m * h * w);
        inputGrads = Matrices.createF(channels * h * w, m);
        batch = m;
    }
}
