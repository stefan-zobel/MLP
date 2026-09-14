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
package math.ml.mlp;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

/** Turns each column of logits into a probability distribution. */
public final class Softmax extends AbstractLayer {

    private MatrixF output;

    /** Creates a softmax layer. */
    public Softmax() {
    }

    public MatrixF forward(MatrixF input) {
        // no need to remember the input
        MatrixF output = Matrices.sameDimF(input);
        float[] in = input.getArrayUnsafe();
        float[] out = output.getArrayUnsafe();
        // compute softmax for each column of input
        int length = input.numRows();
        int off = 0;
        for (int col = 0; col < input.numColumns(); ++col) {
            math.dl.Softmax.softmaxF(length, off, in, off, out);
            off += length;
        }
        if (mode == NetworkMode.TRAIN) {
            // but we need the output for training
            this.output = output;
        }
        return output;
    }

    @Override
    public MatrixF backward(MatrixF lossGrads) {
        if (mode == NetworkMode.INFER) {
            return null;
        }
        int rows = output.numRows();
        MatrixF gradientsOut = Matrices.createF(rows, output.numColumns());
        float[] s = output.getArrayUnsafe();
        float[] g = lossGrads.getArrayUnsafe();
        float[] out = gradientsOut.getArrayUnsafe();
        // The Jacobian of one softmax column is diag(s) - s*s^T, and multiplying it out
        // leaves s_i * (g_i - sum_j g_j s_j): one dot product and one pass per column
        // instead of an explicit rows x rows matrix and a gemv. The sum is the only
        // reduction here and stays in double, as the other reductions do.
        for (int off = 0; off < out.length; off += rows) {
            double sum = 0.0;
            for (int i = 0; i < rows; ++i) {
                sum += (double) g[off + i] * s[off + i];
            }
            float shift = (float) sum;
            for (int i = 0; i < rows; ++i) {
                out[off + i] = s[off + i] * (g[off + i] - shift);
            }
        }
        output = null;
        return gradientsOut;
    }
}
