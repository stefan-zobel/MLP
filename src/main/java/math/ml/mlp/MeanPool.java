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
 * Averages each block of consecutive columns into a single column. Over the layout
 * {@link Conv2D} produces this is global average pooling; over a sequence laid out as one
 * block of tokens per sample it is the pooling a classifier head needs.
 */
public class MeanPool extends AbstractLayer {

    private final int blockSize;

    private MatrixF output;
    private MatrixF inputGrads;
    /** The shape the buffers were sized for, or -1 before the first pass. */
    private int features = -1;
    private int batch = -1;

    /**
     * Creates a pooling layer.
     *
     * @param blockSize how many consecutive columns make one block
     */
    public MeanPool(int blockSize) {
        if (blockSize <= 0) {
            throw new IllegalArgumentException("blockSize " + blockSize + " must be positive");
        }
        this.blockSize = blockSize;
    }

    @Override
    public MatrixF forward(MatrixF in) {
        int cols = in.numColumns();
        if (cols % blockSize != 0) {
            throw new IllegalArgumentException(
                    cols + " columns is not a whole number of blocks of " + blockSize);
        }
        int d = in.numRows();
        int m = cols / blockSize;
        ensureBuffers(d, m);
        float[] src = in.getArrayUnsafe();
        float[] dst = output.getArrayUnsafe();
        float scale = 1.0f / blockSize;
        for (int s = 0; s < m; ++s) {
            int base = s * d;
            Arrays.fill(dst, base, base + d, 0.0f);
            int to = (s + 1) * blockSize;
            // walking the source contiguously rather than the destination, so the
            // accumulation reads one cache line after the other
            for (int c = s * blockSize; c < to; ++c) {
                int from = c * d;
                for (int r = 0; r < d; ++r) {
                    dst[base + r] += src[from + r];
                }
            }
            for (int r = 0; r < d; ++r) {
                dst[base + r] *= scale;
            }
        }
        return output;
    }

    @Override
    public MatrixF backward(MatrixF outputGrads) {
        if (mode == NetworkMode.INFER) {
            return null;
        }
        int d = outputGrads.numRows();
        int m = outputGrads.numColumns();
        float[] src = outputGrads.getArrayUnsafe();
        float[] dst = inputGrads.getArrayUnsafe();
        float scale = 1.0f / blockSize;
        for (int s = 0; s < m; ++s) {
            int base = s * d;
            int to = (s + 1) * blockSize;
            for (int c = s * blockSize; c < to; ++c) {
                int into = c * d;
                for (int r = 0; r < d; ++r) {
                    dst[into + r] = src[base + r] * scale;
                }
            }
        }
        return inputGrads;
    }

    private void ensureBuffers(int d, int m) {
        if (features == d && batch == m) {
            return;
        }
        output = Matrices.createF(d, m);
        inputGrads = Matrices.createF(d, m * blockSize);
        features = d;
        batch = m;
    }
}
