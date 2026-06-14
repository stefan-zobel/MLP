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
 * Batch Normalization layer (Ioffe &amp; Szegedy, 2015).
 *
 * <p>Normalizes each feature (row) across the mini-batch dimension (columns),
 * then applies learnable per-feature scale {@code ?} and shift {@code ?}.
 * This stabilizes and accelerates training, especially in deeper networks.
 * Typical placement is <em>after</em> a linear layer and <em>before</em> the
 * activation: {@code Hidden ? BatchNorm ? ReLU}.
 *
 * <h3>Forward pass (TRAIN)</h3>
 * For each feature {@code j} and batch index {@code i}:
 * <pre>
 *   ?_j     = (1/m) ?_i x_{ji}
 *   ?²_j    = (1/m) ?_i (x_{ji} ? ?_j)²
 *   x?_{ji}  = (x_{ji} ? ?_j) / ?(?²_j + ?)
 *   y_{ji}  = ?_j · x?_{ji} + ?_j
 * </pre>
 * Running statistics are updated for use during inference:
 * <pre>
 *   runningMean ? (1?m) · runningMean + m · ?_batch
 *   runningVar  ? (1?m) · runningVar  + m · ?²_batch
 * </pre>
 * where {@code m} is {@code momentum} (default 0.1).
 *
 * <h3>Forward pass (INFER)</h3>
 * Uses the accumulated running statistics:
 * <pre>
 *   y_{ji} = ?_j · (x_{ji} ? runningMean_j) / ?(runningVar_j + ?) + ?_j
 * </pre>
 *
 * <h3>Backward pass</h3>
 * Full analytic gradient through all three statistics.  Updates {@code ?} and
 * {@code ?} in place (averaged over the batch, consistent with
 * {@link Hidden#backward}).
 */
public class BatchNorm extends AbstractLayer {

    private final int     features;
    private final float   eps;
    /** Weight for new batch statistics in the running-average update. */
    private final float   momentum;

    // Learnable parameters (features × 1)
    private final MatrixF gamma;       // scale,  initialized to 1
    private final MatrixF beta;        // shift,  initialized to 0

    // Running statistics for inference (features × 1)
    private final MatrixF runningMean; // initialized to 0
    private final MatrixF runningVar;  // initialized to 1

    // Cached during TRAIN forward for use in backward
    private MatrixF xHat;      // normalized input    (features × batchSize)
    private MatrixF batchMean; // per-feature mean    (features × 1)
    private MatrixF invStd;    // 1/?(var+?) per row  (features × 1)

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    /**
     * Creates a BatchNorm layer with default {@code ? = 1e-5} and
     * {@code momentum = 0.1}.
     *
     * @param features number of input features (= number of rows of the input
     *                 matrix)
     */
    public BatchNorm(int features) {
        this(features, 1e-5f, 0.1f);
    }

    /**
     * Creates a BatchNorm layer with explicit numerical parameters.
     *
     * @param features number of input features
     * @param eps      small constant added to the variance for numerical
     *                 stability (e.g. {@code 1e-5f})
     * @param momentum weight applied to the <em>new</em> batch statistics
     *                 when updating the running statistics; the old value gets
     *                 weight {@code (1 - momentum)}; typical value: {@code 0.1f}
     */
    public BatchNorm(int features, float eps, float momentum) {
        this.features = features;
        this.eps      = eps;
        this.momentum = momentum;

        gamma       = Matrices.createF(features, 1);
        beta        = Matrices.createF(features, 1);
        runningMean = Matrices.createF(features, 1);
        runningVar  = Matrices.createF(features, 1);

        // ? = 1, runningVar = 1; ? and runningMean stay 0
        for (int i = 0; i < features; i++) {
            gamma.setUnsafe(i, 0, 1.0f);
            runningVar.setUnsafe(i, 0, 1.0f);
        }
    }

    // -------------------------------------------------------------------------
    // Layer contract
    // -------------------------------------------------------------------------

    @Override
    public MatrixF forward(MatrixF input) {
        super.forward(input); // caches this.input when mode == TRAIN

        int d = input.numRows();
        int m = input.numColumns();

        if (mode == NetworkMode.INFER) {
            return applyAffine(input, runningMean, runningVar, d, m);
        }

        // --- TRAIN ---

        // 1. Batch mean per feature (row)
        batchMean = Matrices.createF(d, 1);
        for (int r = 0; r < d; r++) {
            float sum = 0.0f;
            for (int c = 0; c < m; c++) {
                sum += input.getUnsafe(r, c);
            }
            batchMean.setUnsafe(r, 0, sum / m);
        }

        // 2. Batch variance + inverse std-dev per feature
        invStd = Matrices.createF(d, 1);
        MatrixF batchVar = Matrices.createF(d, 1);
        for (int r = 0; r < d; r++) {
            float mean = batchMean.getUnsafe(r, 0);
            float var  = 0.0f;
            for (int c = 0; c < m; c++) {
                float diff = input.getUnsafe(r, c) - mean;
                var += diff * diff;
            }
            var /= m;
            batchVar.setUnsafe(r, 0, var);
            invStd.setUnsafe(r, 0, 1.0f / (float) Math.sqrt(var + eps));
        }

        // 3. Normalize and apply ?/?; cache x? for backward
        xHat = Matrices.createF(d, m);
        MatrixF output = Matrices.createF(d, m);
        for (int r = 0; r < d; r++) {
            float mean = batchMean.getUnsafe(r, 0);
            float iStd = invStd.getUnsafe(r, 0);
            float g    = gamma.getUnsafe(r, 0);
            float b    = beta.getUnsafe(r, 0);
            for (int c = 0; c < m; c++) {
                float xn = (input.getUnsafe(r, c) - mean) * iStd;
                xHat.setUnsafe(r, c, xn);
                output.setUnsafe(r, c, g * xn + b);
            }
        }

        // 4. Update running statistics (exponential moving average)
        float keepWeight = 1.0f - momentum;
        for (int r = 0; r < d; r++) {
            runningMean.setUnsafe(r, 0,
                    keepWeight * runningMean.getUnsafe(r, 0) + momentum * batchMean.getUnsafe(r, 0));
            runningVar.setUnsafe(r, 0,
                    keepWeight * runningVar.getUnsafe(r, 0)  + momentum * batchVar.getUnsafe(r, 0));
        }

        return output;
    }

    /**
     * Backpropagates through the full batch-normalization computation:
     * updates {@code ?} and {@code ?} (averaged over the batch) and returns
     * the gradient w.r.t. the input.
     *
     * <p>Derivation (per feature {@code r}):
     * <pre>
     *   dL/dx?[r,c]  = dLdy[r,c] · ?_r
     *   dL/d?²_r    = ?_c dL/dx?[r,c] · (x[r,c]??_r) · (?½) · invStd³
     *   dL/d?_r     = ?invStd · ?_c dL/dx?[r,c]       (sum(x??)=0 term vanishes)
     *   dL/dx[r,c]  = dL/dx?[r,c]·invStd
     *               + dL/d?²_r · 2·(x[r,c]??_r)/m
     *               + dL/d?_r / m
     * </pre>
     */
    @Override
    public MatrixF backward(MatrixF dLdy, float learningRate) {
        if (mode == NetworkMode.INFER) {
            return null;
        }
        int d = dLdy.numRows();
        int m = dLdy.numColumns();
        MatrixF dLdx = Matrices.createF(d, m);

        for (int r = 0; r < d; r++) {
            float g    = gamma.getUnsafe(r, 0);
            float iStd = invStd.getUnsafe(r, 0);
            float mean = batchMean.getUnsafe(r, 0);

            // --- ? and ? updates (averaged over batch) ---
            float dGamma = 0.0f, dBeta = 0.0f;
            for (int c = 0; c < m; c++) {
                float dl = dLdy.getUnsafe(r, c);
                dGamma += dl * xHat.getUnsafe(r, c);
                dBeta  += dl;
            }
            gamma.setUnsafe(r, 0, g                        - learningRate * dGamma / m);
            beta.setUnsafe(r, 0,  beta.getUnsafe(r, 0)     - learningRate * dBeta  / m);

            // --- dL/d?²_r ---
            float dVar = 0.0f;
            for (int c = 0; c < m; c++) {
                float dxhat = dLdy.getUnsafe(r, c) * g;
                dVar += dxhat * (input.getUnsafe(r, c) - mean);
            }
            dVar *= -0.5f * iStd * iStd * iStd; // multiply by ?½ · invStd³

            // --- dL/d?_r  (the variance-path term vanishes: ?(x??)=0) ---
            float dMean = 0.0f;
            for (int c = 0; c < m; c++) {
                dMean += dLdy.getUnsafe(r, c) * g;
            }
            dMean *= -iStd;

            // --- dL/dx[r,c] ---
            for (int c = 0; c < m; c++) {
                float dxhat      = dLdy.getUnsafe(r, c) * g;
                float xMinusMean = input.getUnsafe(r, c) - mean;
                dLdx.setUnsafe(r, c,
                        dxhat * iStd
                        + dVar * 2.0f * xMinusMean / m
                        + dMean / m);
            }
        }

        // Release cached state
        xHat      = null;
        batchMean = null;
        invStd    = null;
        input     = null; // inherited from AbstractLayer

        return dLdx;
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /**
     * Applies the affine transform {@code y = ?·(x?mean)/?(var+?) + ?}
     * using the supplied statistics (used for inference with running stats).
     */
    private MatrixF applyAffine(MatrixF x, MatrixF mean, MatrixF var, int d, int m) {
        MatrixF out = Matrices.createF(d, m);
        for (int r = 0; r < d; r++) {
            float mu   = mean.getUnsafe(r, 0);
            float iStd = 1.0f / (float) Math.sqrt(var.getUnsafe(r, 0) + eps);
            float g    = gamma.getUnsafe(r, 0);
            float b    = beta.getUnsafe(r, 0);
            for (int c = 0; c < m; c++) {
                out.setUnsafe(r, c, g * (x.getUnsafe(r, c) - mu) * iStd + b);
            }
        }
        return out;
    }
}
