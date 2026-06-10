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
 * <h3>Loss per sample</h3>
 * <pre>
 *   L = ?(1 / dim) · ?_i [ t_i · log(p_i) + (1 ? t_i) · log(1 ? p_i) ]
 * </pre>
 * where {@code dim} is the number of output elements (e.g. 784 for MNIST),
 * {@code t} is the reconstruction target (the original input image, values in
 * [0, 1]) and {@code p} is the sigmoid-activated decoder prediction.
 *
 * <h3>Gradient returned by {@link #forward}</h3>
 * <pre>
 *   ?L/?p_i = (p_i ? t_i) / [ p_i · (1 ? p_i) ]
 * </pre>
 * <b>Note on numerical stability:</b> when this layer is immediately preceded
 * by a {@link Sigmoid} layer the backward pass automatically computes the
 * combined gradient at the sigmoid <em>input</em>:
 * <pre>
 *   ?L/?x_i = ?L/?p_i · ?'(x_i)
 *            = (p_i ? t_i) / [p·(1?p)] · p·(1?p)
 *            = p_i ? t_i
 * </pre>
 * So the p·(1?p) terms cancel and the result is perfectly well-behaved.
 *
 * <h3>Usage in a VAE network</h3>
 * <pre>{@code
 * BinaryCrossEntropyLoss bce = new BinaryCrossEntropyLoss();
 * bce.registerLossCallback(net::onLossComputationCompleted);
 * // Provide the original input images as reconstruction targets:
 * bce.registerBatchExpectedValuesProvider(net::getExpectedBatchResults);
 * net.add(bce);
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
     *                   shape {@code dim × batchSize}
     * @return gradient ?L/?prediction, same shape as {@code prediction}
     */
    @Override
    public MatrixF forward(MatrixF prediction) {
        MatrixF expected = getExpectation();
        if (expected == null) {
            return null;
        }
        computeLosses(prediction, expected);
        return computeGradients(prediction, expected);
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    private MatrixF computeGradients(MatrixF pred, MatrixF expect) {
        int rows = pred.numRows();
        int cols = pred.numColumns();
        MatrixF gradients = Matrices.createF(rows, cols);
        for (int c = 0; c < cols; c++) {
            for (int r = 0; r < rows; r++) {
                float p = pred.getUnsafe(r, c);
                float t = expect.getUnsafe(r, c);
                // Gradient: (p - t) / (p * (1 - p)).
                // The denominator is clamped to avoid division by zero at the
                // extremes p = 0 and p = 1.  When Sigmoid precedes this layer
                // the Sigmoid backward multiplies by p*(1-p), so the terms
                // cancel and the combined gradient at the sigmoid input is
                // simply p - t (see class Javadoc).
                float denom = clamp(p) * clamp(1.0f - p);
                gradients.setUnsafe(r, c, (p - t) / denom);
            }
        }
        return gradients;
    }

    /**
     * Computes the per-sample BCE loss (normalised by {@code dim}) and
     * delivers it to the registered loss callback.
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
                // Normalise by dim so the loss is comparable across different
                // output sizes (e.g. 784 pixels for MNIST).
                loss.setUnsafe(0, c, sum / rows);
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
