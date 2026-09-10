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
 * Element-wise Binary Cross-Entropy (BCE) reconstruction loss.
 *
 * <p>Suitable as the reconstruction term in a Variational Autoencoder when the
 * final decoder activation is {@link Sigmoid} (output values in (0, 1)).
 *
 * <h2>Loss per sample</h2>
 * <pre>
 *   L = &minus;&sum;_i [ t_i &middot; log(p_i) + (1 &minus; t_i) &middot; log(1 &minus; p_i) ]
 * </pre>
 * summed over the output elements, where {@code t} is the reconstruction target
 * (the original input image, values in [0, 1]) and {@code p} is the
 * sigmoid-activated decoder prediction.
 *
 * <h2>Gradient returned by {@link #forward}</h2>
 * <pre>
 *   &part;L/&part;p_i = (p_i &minus; t_i) / [ p_i &middot; (1 &minus; p_i) ]
 * </pre>
 * <b>Note on numerical stability:</b> when this layer is immediately preceded
 * by a {@link Sigmoid} layer the backward pass automatically computes the
 * combined gradient at the sigmoid <em>input</em>:
 * <pre>
 *   &part;L/&part;x_i = &part;L/&part;p_i &middot; &sigma;'(x_i)
 *            = (p_i &minus; t_i) / [p&middot;(1&minus;p)] &middot; p&middot;(1&minus;p)
 *            = p_i &minus; t_i
 * </pre>
 * The p&middot;(1&minus;p) terms cancel in exact arithmetic, but not in {@code float}:
 * from about {@code x = 17} the sigmoid rounds to exactly {@code 1.0f}, its
 * derivative becomes {@code 0}, and the gradient is lost. Prefer
 * {@link SigmoidBCELoss}, which fuses the two and stays exact.
 *
 * <h2>Usage in a VAE network</h2>
 * <pre>{@code
 * BinaryCrossEntropyLoss bce = new BinaryCrossEntropyLoss();
 * bce.registerLossCallback(net::onLossComputationCompleted);
 * net.add(bce);
 * // the training loop passes the original images as reconstruction targets:
 * net.train(images, images, learningRate);
 * }</pre>
 */
public class BinaryCrossEntropyLoss extends AbstractLoss {

    public BinaryCrossEntropyLoss() {
    }

    /**
     * Computes the BCE reconstruction loss (fires the loss callback if
     * registered) and returns the gradient w.r.t. the predictions.
     *
     * @param prediction sigmoid-activated decoder output (values in (0, 1)),
     *                   shape {@code dim x batchSize}
     * @return gradient &part;L/&part;prediction, same shape as {@code prediction}
     */
    @Override
    public MatrixF forward(MatrixF prediction) {
        MatrixF expected = getExpectation();
        computeLosses(prediction, expected);
        return computeGradients(prediction, expected);
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /**
     * Gradient {@code (p - t) / (p * (1 - p))}, with the denominator clamped so
     * that {@code p = 0} and {@code p = 1} cannot divide by zero.
     */
    private MatrixF computeGradients(MatrixF pred, MatrixF expect) {
        MatrixF denominator = pred.map(BinaryCrossEntropyLoss::clamp)
                .hadamard(pred.map(p -> clamp(1.0f - p)));
        return MatrixOps.divInplace(pred.minus(expect), denominator);
    }

    /**
     * Computes the per-sample BCE loss and delivers it to the registered loss
     * callback.
     *
     * <p>Deliberately a loop: the matrix-API form needs five {@code dim x batch}
     * temporaries where this needs one, and it only reports a number.
     */
    private void computeLosses(MatrixF pred, MatrixF expect) {
        if (lossCallback != null) {
            int rows = pred.numRows();
            int cols = pred.numColumns();
            MatrixF loss = Matrices.createF(1, cols);
            for (int c = 0; c < cols; c++) {
                float sum = 0.0f;
                for (int r = 0; r < rows; r++) {
                    float p = pred.getUnsafe(r, c);
                    float t = expect.getUnsafe(r, c);
                    sum -= t * log(p) + (1.0f - t) * log(1.0f - p);
                }
                // summed over the output dimension, so it agrees with the
                // summed gradient this layer returns
                loss.setUnsafe(0, c, sum);
            }
            lossCallback.accept(loss);
        }
    }

    private static float log(float x) {
        return (float) Math.log(clamp(x));
    }

    private static float clamp(float x) {
        return Math.max(x, Float.MIN_NORMAL);
    }
}
