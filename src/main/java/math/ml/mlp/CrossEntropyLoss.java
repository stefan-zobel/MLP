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

/**
 * The smaller the cross-entropy, the more similar the two probability
 * distributions are.
 */
public class CrossEntropyLoss extends AbstractLoss {

    public CrossEntropyLoss() {
    }

    // compute losses
    // compute accuracy
    // compute gradients
    // return gradients
    @Override
    public MatrixF forward(MatrixF prediction) {
        MatrixF expected = getExpectation();
        if (expected == null) {
            return null;
        }
        computeLosses(prediction, expected);
        computeAccuracy(prediction, expected);
        return computeGradients(prediction, expected);
    }

    private MatrixF computeGradients(MatrixF pred, MatrixF expect) {
        MatrixF gradients = Matrices.createF(pred.numRows(), pred.numColumns());
        for (int i = 0, colEnd = pred.endCol(), rowEnd = pred.endRow(); i <= colEnd; ++i) {
            for (int j = 0; j <= rowEnd; ++j) {
                float expected = expect.getUnsafe(j, i);
                if (expected != 0.0f) {
                    gradients.set(j, i, (-expected / clamp(pred.getUnsafe(j, i))));
                }
            }
        }
        return gradients;
    }

    /**
     * Computes 1 x batchSize row vector of cross-entropy losses for same-sized
     * matrices of expected values (expect) and predicted values (pred) expressed as
     * column vectors in those matrices.
     * 
     * @param pred   predicted values as a matrix of column vectors
     * @param expect expected values as a matrix of column vectors
     */
    private void computeLosses(MatrixF pred, MatrixF expect) {
        if (lossCallback != null) {
            MatrixF loss = Matrices.createF(1, pred.numColumns());
            for (int i = 0, colEnd = pred.endCol(), rowEnd = pred.endRow(); i <= colEnd; ++i) {
                float productSum = 0.0f;
                for (int j = 0; j <= rowEnd; ++j) {
                    float expected = expect.getUnsafe(j, i);
                    if (expected != 0.0f) {
                        productSum += expected * log(pred.getUnsafe(j, i));
                    }
                }
                loss.setUnsafe(0, i, -productSum);
            }
            lossCallback.accept(loss);
        }
    }

//    private void computeAccuracy(MatrixF pred, MatrixF expect) {
//        if (accuracyCallback != null) {
//            accuracyCallback.accept(CategorialHellingerAccuracy.computeAccuracy(pred, expect));
//        }
//    }

    private void computeAccuracy(MatrixF pred, MatrixF expect) {
        if (accuracyCallback != null) {
            accuracyCallback.accept(CategorialAccuracy.computeAccuracy(pred, expect));
        }
    }

    private static float log(float x) {
        return (float) Math.log(clamp(x));
    }

    private static float clamp(float x) {
        // x should never be <= 0
        if (x <= 0.0f) {
            return Float.MIN_NORMAL;
        }
        return x;
    }
}
