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

public final class Softmax extends AbstractLayer {

    private MatrixF output;

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
    public MatrixF backward(MatrixF lossGrads, float unused) {
        if (mode == NetworkMode.INFER) {
            return null;
        }
        int rows = output.numRows();
        int cols = output.numColumns();
        MatrixF gradientsOut = Matrices.createF(rows, cols);
        MatrixF jacobian = Matrices.createF(rows, rows);
        MatrixF oneGrad = Matrices.createF(rows, 1);
        for (int col = 0; col < cols; ++col) {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < rows; ++j) {
                    if (i < j) {
                        float si = output.getUnsafe(i, col);
                        float sj = output.getUnsafe(j, col);
                        float sij = -si * sj;
                        jacobian.setUnsafe(j, i, sij);
                        jacobian.setUnsafe(i, j, sij);
                    } else if (i == j) {
                        float sii = output.getUnsafe(i, col);
                        jacobian.setUnsafe(i, i, sii * (1.0f - sii));
                    }
                }
            }
            // get the corresponding lossGrads column
            MatrixF lossGrad = lossGrads.selectColumn(col);
            // compute the gradient for this column
            oneGrad = jacobian.mult(lossGrad, oneGrad);
            // store it in the corresponding column of gradientsOut
            gradientsOut.setColumnInplace(col, oneGrad);
        }
        output = null;
        return gradientsOut;
    }
}
