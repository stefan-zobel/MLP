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
 * Sigmoid combined with binary cross-entropy, for a decoder that reconstructs
 * values in (0, 1).
 *
 * <p>Fusing the two yields the gradient {@code p - t} directly and therefore
 * keeps it exact at saturated logits, where a separate {@link Sigmoid} followed
 * by {@link BinaryCrossEntropyLoss} loses it to float underflow.
 */
public class SigmoidBCELoss extends AbstractLoss {

    private MatrixF gradients;

    /** Expects logits, not probabilities: the sigmoid is applied here. */
    @Override
    public MatrixF forward(MatrixF logits) {
        MatrixF probabilities = Matrices.sameDimF(logits);
        float[] in = logits.getArrayUnsafe();
        float[] out = probabilities.getArrayUnsafe();
        for (int i = 0; i < in.length; ++i) {
            out[i] = sigmoid(in[i]);
        }
        if (mode == NetworkMode.INFER) {
            return probabilities;
        }
        MatrixF expected = getExpectation();
        computeLosses(logits, expected);
        gradients = probabilities.minus(expected);
        return gradients;
    }

    @Override
    public MatrixF backward(MatrixF unused1, float unused2) {
        if (mode == NetworkMode.INFER) {
            return null;
        }
        MatrixF gradsOut = gradients;
        gradients = null;
        return gradsOut;
    }

    @Override
    public boolean producesPredictionInInferMode() {
        return true;
    }

    /**
     * Computes the loss from the logits as
     * {@code max(x, 0) - x * t + log1p(exp(-|x|))}, a form that cannot reach
     * {@code log(0)}.
     *
     * <p>The loss is summed over the output dimension so that it agrees with the
     * summed gradient this layer returns.
     */
    private void computeLosses(MatrixF logits, MatrixF expect) {
        if (lossCallback != null) {
            int rows = logits.numRows();
            int cols = logits.numColumns();
            MatrixF loss = Matrices.createF(1, cols);
            for (int c = 0; c < cols; ++c) {
                double sum = 0.0;
                for (int r = 0; r < rows; ++r) {
                    double x = logits.getUnsafe(r, c);
                    double t = expect.getUnsafe(r, c);
                    sum += Math.max(x, 0.0) - x * t + Math.log1p(Math.exp(-Math.abs(x)));
                }
                loss.setUnsafe(0, c, (float) sum);
            }
            lossCallback.accept(loss);
        }
    }

    private static float sigmoid(float x) {
        return 1.0f / (1.0f + (float) Math.exp(-x));
    }
}
