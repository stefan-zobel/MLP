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

import java.util.Arrays;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

/**
 * Max pooling over the same {@code channels x (batch * height * width)} layout
 * {@link Conv2D} produces. A window that does not fit into the input is dropped rather
 * than padded.
 */
public class MaxPool2D extends AbstractLayer {

    private final int channels;
    private final int inH;
    private final int inW;
    private final int window;
    private final int stride;
    private final int outH;
    private final int outW;

    /** Where each output element came from, as an index into the input array. */
    private int[] argmax;
    private MatrixF output;
    private MatrixF inputGrads;
    /** The batch size the buffers above were sized for, or -1 before the first pass. */
    private int batch = -1;

    /**
     * Creates a pooling layer whose stride equals its window, the non-overlapping case.
     *
     * @param channels channels of the input, unchanged by pooling
     * @param inH      height of the input
     * @param inW      width of the input
     * @param window   edge length of the square pooling window
     */
    public MaxPool2D(int channels, int inH, int inW, int window) {
        this(channels, inH, inW, window, window);
    }

    /**
     * Creates a pooling layer.
     *
     * @param channels channels of the input, unchanged by pooling
     * @param inH      height of the input
     * @param inW      width of the input
     * @param window   edge length of the square pooling window
     * @param stride   step between neighboring windows
     */
    public MaxPool2D(int channels, int inH, int inW, int window, int stride) {
        if (channels <= 0 || window <= 0 || stride <= 0) {
            throw new IllegalArgumentException(
                    "channels " + channels + ", window " + window + " and stride " + stride + " must all be positive");
        }
        this.channels = channels;
        this.inH = inH;
        this.inW = inW;
        this.window = window;
        this.stride = stride;
        int spanH = inH - window;
        int spanW = inW - window;
        // the span has to be tested before the division: integer division truncates
        // towards zero, so a window that does not fit at all still divides to a
        // positive count once the stride exceeds the shortfall
        if (spanH < 0 || spanW < 0) {
            throw new IllegalArgumentException("a " + window + "x" + window + " window does not fit into a " + inH + "x"
                    + inW + " input");
        }
        this.outH = spanH / stride + 1;
        this.outW = spanW / stride + 1;
    }

    /** Height of this layer's output. */
    public int outputHeight() {
        return outH;
    }

    /** Width of this layer's output. */
    public int outputWidth() {
        return outW;
    }

    @Override
    public MatrixF forward(MatrixF in) {
        if (in.numRows() != channels) {
            throw new IllegalArgumentException("expected " + channels + " channels, got " + in.numRows());
        }
        int inSpatial = inH * inW;
        if (in.numColumns() % inSpatial != 0) {
            throw new IllegalArgumentException(
                    in.numColumns() + " columns is not a whole number of " + inH + "x" + inW + " images");
        }
        int m = in.numColumns() / inSpatial;
        ensureBuffers(m);
        float[] src = in.getArrayUnsafe();
        float[] dst = output.getArrayUnsafe();
        int outSpatial = outH * outW;
        for (int s = 0; s < m; ++s) {
            int sample = s * inSpatial * channels;
            for (int oy = 0; oy < outH; ++oy) {
                for (int ox = 0; ox < outW; ++ox) {
                    int column = (s * outSpatial + oy * outW + ox) * channels;
                    int iy0 = oy * stride;
                    int ix0 = ox * stride;
                    for (int c = 0; c < channels; ++c) {
                        int bestIdx = -1;
                        float best = 0.0f;
                        for (int wy = 0; wy < window; ++wy) {
                            int rowBase = sample + (iy0 + wy) * inW * channels;
                            for (int wx = 0; wx < window; ++wx) {
                                int idx = rowBase + (ix0 + wx) * channels + c;
                                float v = src[idx];
                                // strictly greater, so a tie keeps the lowest index
                                if (bestIdx < 0 || v > best) {
                                    best = v;
                                    bestIdx = idx;
                                }
                            }
                        }
                        dst[column + c] = best;
                        argmax[column + c] = bestIdx;
                    }
                }
            }
        }
        return output;
    }

    @Override
    public MatrixF backward(MatrixF outputGrads) {
        if (mode == NetworkMode.INFER) {
            return null;
        }
        float[] src = outputGrads.getArrayUnsafe();
        float[] dst = inputGrads.getArrayUnsafe();
        Arrays.fill(dst, 0.0f);
        // every position that did not win its window contributes nothing, which is why
        // the destination is cleared rather than written element by element
        for (int i = 0; i < src.length; ++i) {
            dst[argmax[i]] += src[i];
        }
        return inputGrads;
    }

    private void ensureBuffers(int m) {
        if (batch == m) {
            return;
        }
        output = Matrices.createF(channels, m * outH * outW);
        inputGrads = Matrices.createF(channels, m * inH * inW);
        argmax = new int[channels * m * outH * outW];
        batch = m;
    }
}
