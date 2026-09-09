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

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

/**
 * Batch Normalization layer (Ioffe &amp; Szegedy, 2015).
 *
 * <p>Normalizes each feature (row) across the mini-batch dimension (columns),
 * then applies learnable per-feature scale {@code gamma} and shift {@code beta}.
 * This stabilizes and accelerates training, especially in deeper networks.
 * Typical placement is <em>after</em> a linear layer and <em>before</em> the
 * activation: {@code Hidden -> BatchNorm -> ReLU}.
 *
 * <h2>Forward pass (TRAIN)</h2>
 * For each feature {@code j} and batch index {@code i}:
 * <pre>
 *   &mu;_j     = (1/m) &sum;_i x_{ji}
 *   &sigma;&sup2;_j    = (1/m) &sum;_i (x_{ji} &minus; &mu;_j)&sup2;
 *   x&#770;_{ji}  = (x_{ji} &minus; &mu;_j) / &radic;(&sigma;&sup2;_j + &epsilon;)
 *   y_{ji}  = &gamma;_j &middot; x&#770;_{ji} + &beta;_j
 * </pre>
 * Running statistics are updated for use during inference:
 * <pre>
 *   runningMean &larr; (1&minus;m) &middot; runningMean + m &middot; &mu;_batch
 *   runningVar  &larr; (1&minus;m) &middot; runningVar  + m &middot; &sigma;&sup2;_batch
 * </pre>
 * where {@code m} is {@code momentum} (default 0.1).
 *
 * <h2>Forward pass (INFER)</h2>
 * Uses the accumulated running statistics:
 * <pre>
 *   y_{ji} = &gamma;_j &middot; (x_{ji} &minus; runningMean_j) / &radic;(runningVar_j + &epsilon;) + &beta;_j
 * </pre>
 *
 * <h2>Backward pass</h2>
 * Full analytic gradient through all three statistics.  Updates {@code gamma} and
 * {@code beta} in place (averaged over the batch, consistent with
 * {@link Hidden#backward}).
 */
public class BatchNorm extends AbstractLayer {

    private static final String LOAD_DIR = "./data/";
    private static final String STORE_DIR = "./checkpoints/";

    private final int     features;
    private final String  name;
    private final boolean storeParameters;
    private final float   eps;
    /** Weight for new batch statistics in the running-average update. */
    private final float   momentum;

    // Learnable parameters (features x 1)
    private final MatrixF gamma;       // scale,  initialized to 1
    private final MatrixF beta;        // shift,  initialized to 0

    // Running statistics for inference (features x 1)
    private final MatrixF runningMean; // initialized to 0
    private final MatrixF runningVar;  // initialized to 1

    // Cached during TRAIN forward for use in backward
    private MatrixF xHat;      // normalized input    (features x batchSize)
    private MatrixF batchMean; // per-feature mean    (features x 1)
    private MatrixF invStd;    // 1/sqrt(var+eps) per row  (features x 1)

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    /**
     * Creates a BatchNorm layer with default {@code eps = 1e-5} and
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
        this(features, eps, momentum, null, false, false);
    }

    /**
     * Creates a BatchNorm layer that can persist its parameters, mirroring the
     * {@link Hidden} constructor.
     *
     * @param features number of input features
     * @param name     identifies the checkpoint file {@code bn_<name>}
     * @param load     read the parameters from {@code ./data/} at construction
     * @param store    let {@link #storeParameters()} write to {@code ./checkpoints/}
     */
    public BatchNorm(int features, String name, boolean load, boolean store) {
        this(features, 1e-5f, 0.1f, name, load, store);
    }

    /**
     * Creates a persisting BatchNorm layer with explicit numerical parameters.
     *
     * @param features number of input features
     * @param eps      small constant added to the variance for numerical stability
     * @param momentum weight applied to the new batch statistics
     * @param name     identifies the checkpoint file {@code bn_<name>}
     * @param load     read the parameters from {@code ./data/} at construction
     * @param store    let {@link #storeParameters()} write to {@code ./checkpoints/}
     */
    public BatchNorm(int features, float eps, float momentum, String name, boolean load, boolean store) {
        this.features = features;
        this.eps      = eps;
        this.momentum = momentum;
        this.name     = name;
        this.storeParameters = store;

        // gamma = 1, runningVar = 1; beta and runningMean stay 0
        gamma       = Matrices.onesF(features, 1);
        beta        = Matrices.createF(features, 1);
        runningMean = Matrices.createF(features, 1);
        runningVar  = Matrices.onesF(features, 1);

        if (load) {
            try (FileInputStream fis = new FileInputStream(LOAD_DIR + "bn_" + name)) {
                readParameters(fis);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
    }

    /**
     * Writes gamma, beta, runningMean and runningVar to {@code os} in that
     * order. All four belong together, so they share one file.
     *
     * @param os the stream to write into; the caller closes it
     * @throws IOException if writing fails
     */
    void writeParameters(OutputStream os) throws IOException {
        Matrices.serializeF(gamma, os);
        Matrices.serializeF(beta, os);
        Matrices.serializeF(runningMean, os);
        Matrices.serializeF(runningVar, os);
    }

    /**
     * Reads the four parameter matrices back in the order
     * {@link #writeParameters(OutputStream)} wrote them.
     *
     * @param is the stream to read from; the caller closes it
     * @throws IOException if reading fails
     */
    void readParameters(InputStream is) throws IOException {
        gamma.setInplace(Matrices.deserializeF(is));
        beta.setInplace(Matrices.deserializeF(is));
        runningMean.setInplace(Matrices.deserializeF(is));
        runningVar.setInplace(Matrices.deserializeF(is));
    }

    /**
     * Persists all four parameter matrices if this layer was constructed with
     * storing enabled.
     */
    @Override
    public void storeParameters() {
        if (!storeParameters) {
            return;
        }
        try {
            Files.createDirectories(Paths.get(STORE_DIR));
            try (FileOutputStream fos = new FileOutputStream(STORE_DIR + "bn_" + name)) {
                writeParameters(fos);
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
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
            return applyAffine(input, runningMean, runningVar);
        }

        // --- TRAIN ---

        // 1. Batch mean per feature; colsAverage collapses the columns, so the
        //    result is the d x 1 per-row mean
        batchMean = Matrices.colsAverage(input);

        // 2. Batch variance and inverse standard deviation per feature
        MatrixF centered = input.plusBroadcastedVector(batchMean.uminus());
        MatrixF batchVar = Matrices.colsAverage(centered.hadamard(centered));
        float epsilon = eps;
        invStd = batchVar.map(v -> (float) (1.0 / Math.sqrt(v + epsilon)));

        // 3. Normalize and apply gamma/beta; cache xHat for backward
        xHat = MatrixOps.mulRowsInplace(centered, invStd);
        MatrixF output = MatrixOps.mulRowsInplace(xHat.copy(), gamma).addBroadcastedVectorInplace(beta);

        // 4. Update running statistics (exponential moving average)
        runningMean.scaleInplace(1.0f - momentum).addInplace(momentum, batchMean);
        runningVar.scaleInplace(1.0f - momentum).addInplace(momentum, batchVar);

        return output;
    }

    /**
     * Backpropagates through the full batch-normalization computation:
     * updates {@code gamma} and {@code beta} (averaged over the batch) and returns
     * the gradient w.r.t. the input.
     *
     * <p>Derivation (per feature {@code r}):
     * <pre>
     *   dL/dx&#770;[r,c]  = dLdy[r,c] &middot; &gamma;_r
     *   dL/d&sigma;&sup2;_r    = &sum;_c dL/dx&#770;[r,c] &middot; (x[r,c]&minus;&mu;_r) &middot; (&minus;&frac12;) &middot; invStd&sup3;
     *   dL/d&mu;_r     = &minus;invStd &middot; &sum;_c dL/dx&#770;[r,c]       (sum(x&minus;&mu;)=0 term vanishes)
     *   dL/dx[r,c]  = dL/dx&#770;[r,c]&middot;invStd
     *               + dL/d&sigma;&sup2;_r &middot; 2&middot;(x[r,c]&minus;&mu;_r)/m
     *               + dL/d&mu;_r / m
     * </pre>
     */
    @Override
    public MatrixF backward(MatrixF dLdy, float learningRate) {
        if (mode == NetworkMode.INFER) {
            return null;
        }
        int m = dLdy.numColumns();

        // --- gamma and beta gradients (summed here, averaged in the update) ---
        MatrixF dGamma = Matrices.sumColumns(dLdy.hadamard(xHat));
        MatrixF dBeta = Matrices.sumColumns(dLdy);

        // dxHat needs the OLD gamma, so scale before the parameters move
        MatrixF dxHat = MatrixOps.mulRowsInplace(dLdy.copy(), gamma);

        gamma.addInplace(-learningRate / m, dGamma);
        beta.addInplace(-learningRate / m, dBeta);

        // forward() consumed its centered matrix as xHat, so recompute it here
        MatrixF centered = input.plusBroadcastedVector(batchMean.uminus());

        // --- dL/dvar = -1/2 * invStd^3 * sum_c dxHat * (x - mean) ---
        MatrixF dVar = Matrices.sumColumns(dxHat.hadamard(centered));
        MatrixOps.mulRowsInplace(dVar, invStd);
        MatrixOps.mulRowsInplace(dVar, invStd);
        MatrixOps.mulRowsInplace(dVar, invStd);
        dVar.scaleInplace(-0.5f);

        // --- dL/dmean = -invStd * sum_c dxHat  (the variance path vanishes: sum(x-mean)=0) ---
        MatrixF dMean = MatrixOps.mulRowsInplace(Matrices.sumColumns(dxHat), invStd).scaleInplace(-1.0f);

        // --- dL/dx = dxHat*invStd + dVar*2*(x-mean)/m + dMean/m ---
        MatrixF dLdx = MatrixOps.mulRowsInplace(dxHat, invStd)
                .addInplace(1.0f, MatrixOps.mulRowsInplace(centered.scaleInplace(2.0f / m), dVar))
                .addBroadcastedVectorInplace(dMean.scaleInplace(1.0f / m));

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
     * Applies the affine transform {@code y = gamma*(x-mean)/sqrt(var+eps) + beta}
     * using the supplied statistics (used for inference with running stats).
     */
    private MatrixF applyAffine(MatrixF x, MatrixF mean, MatrixF var) {
        float epsilon = eps;
        MatrixF inverseStd = var.map(v -> (float) (1.0 / Math.sqrt(v + epsilon)));
        MatrixF out = x.plusBroadcastedVector(mean.uminus());
        MatrixOps.mulRowsInplace(out, inverseStd);
        MatrixOps.mulRowsInplace(out, gamma);
        return out.addBroadcastedVectorInplace(beta);
    }
}
