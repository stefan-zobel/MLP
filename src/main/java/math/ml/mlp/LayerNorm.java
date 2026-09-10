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
 * Layer Normalization (Ba, Kiros &amp; Hinton, 2016): normalizes each sample over
 * its features, then applies a learnable per-feature scale and shift.
 *
 * <p>Unlike {@link BatchNorm} this needs no running statistics, because each
 * sample is normalized on its own. TRAIN and INFER therefore behave identically
 * and a restored layer cannot disagree with the one that was saved.
 *
 * <h2>Forward pass</h2>
 * Samples are the columns, so the statistics are taken per column {@code j} over
 * the {@code d} features:
 * <pre>
 *   &mu;_j    = (1/d) &sum;_i x_ij
 *   &sigma;&sup2;_j   = (1/d) &sum;_i (x_ij &minus; &mu;_j)&sup2;
 *   x&#770;_ij  = (x_ij &minus; &mu;_j) / &radic;(&sigma;&sup2;_j + &epsilon;)
 *   y_ij   = &gamma;_i &middot; x&#770;_ij + &beta;_i
 * </pre>
 */
public class LayerNorm extends AbstractLayer {

    private static final String LOAD_DIR = "./data/";
    private static final String STORE_DIR = "./checkpoints/";

    private final int features;
    private final float eps;
    private final String name;
    private final boolean storeParameters;

    // Learnable parameters (features x 1)
    private final MatrixF gamma;
    private final MatrixF beta;

    // Cached during forward for use in backward
    private MatrixF xHat;   // normalized input  (features x batchSize)
    private MatrixF invStd; // 1/sqrt(var+eps)   (1 x batchSize), one per sample

    /**
     * Creates a LayerNorm layer with default {@code eps = 1e-5} that does not
     * persist its parameters.
     *
     * @param features number of input features (= number of rows of the input)
     */
    public LayerNorm(int features) {
        this(features, 1e-5f, null, false, false);
    }

    /**
     * Creates a LayerNorm layer with an explicit epsilon.
     *
     * @param features number of input features
     * @param eps      small constant added to the variance for numerical stability
     */
    public LayerNorm(int features, float eps) {
        this(features, eps, null, false, false);
    }

    /**
     * Creates a LayerNorm layer that can persist its parameters, mirroring the
     * {@link Hidden} and {@link BatchNorm} constructors.
     *
     * @param features number of input features
     * @param name     identifies the checkpoint file {@code ln_<name>}
     * @param load     read the parameters from {@code ./data/} at construction
     * @param store    let {@link #storeParameters()} write to {@code ./checkpoints/}
     */
    public LayerNorm(int features, String name, boolean load, boolean store) {
        this(features, 1e-5f, name, load, store);
    }

    /**
     * Creates a persisting LayerNorm layer with an explicit epsilon.
     *
     * @param features number of input features
     * @param eps      small constant added to the variance for numerical stability
     * @param name     identifies the checkpoint file {@code ln_<name>}
     * @param load     read the parameters from {@code ./data/} at construction
     * @param store    let {@link #storeParameters()} write to {@code ./checkpoints/}
     */
    public LayerNorm(int features, float eps, String name, boolean load, boolean store) {
        this.features = features;
        this.eps = eps;
        this.name = name;
        this.storeParameters = store;

        gamma = Matrices.onesF(features, 1);
        beta = Matrices.createF(features, 1);

        if (load) {
            try (FileInputStream fis = new FileInputStream(LOAD_DIR + "ln_" + name)) {
                readParameters(fis);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
    }

    @Override
    public MatrixF forward(MatrixF input) {
        if (input.numRows() != features) {
            throw new IllegalArgumentException("expected " + features + " features, got " + input.numRows());
        }
        super.forward(input); // caches this.input when mode == TRAIN

        // per-sample statistics: rowsAverage collapses the rows, so the result is
        // the 1 x m mean of each column
        MatrixF mean = Matrices.rowsAverage(input);
        MatrixF centered = MatrixOps.addColumnsInplace(input.copy(), mean.uminus());
        MatrixF variance = Matrices.rowsAverage(centered.hadamard(centered));
        float epsilon = eps;
        invStd = variance.map(v -> (float) (1.0 / Math.sqrt(v + epsilon)));

        xHat = MatrixOps.mulColumnsInplace(centered, invStd);
        return MatrixOps.mulRowsInplace(xHat.copy(), gamma).addBroadcastedVectorInplace(beta);
    }

    /**
     * Backpropagates through the normalization and updates {@code gamma} and
     * {@code beta}, averaged over the batch as {@link Hidden} does.
     */
    @Override
    public MatrixF backward(MatrixF dLdy, float learningRate) {
        if (mode == NetworkMode.INFER) {
            return null;
        }
        int m = dLdy.numColumns();

        MatrixF dGamma = Matrices.sumColumns(dLdy.hadamard(xHat));
        MatrixF dBeta = Matrices.sumColumns(dLdy);

        // dxHat needs the OLD gamma, so scale before the parameters move
        MatrixF dxHat = MatrixOps.mulRowsInplace(dLdy.copy(), gamma);

        gamma.addInplace(-learningRate / m, dGamma);
        beta.addInplace(-learningRate / m, dBeta);

        // dL/dx = invStd * (dxHat - mean_i(dxHat) - xHat * mean_i(dxHat * xHat)),
        // both means taken per sample, i.e. over the features of one column
        MatrixF meanDxHat = Matrices.rowsAverage(dxHat);
        MatrixF meanDxHatXHat = Matrices.rowsAverage(dxHat.hadamard(xHat));

        MatrixF dLdx = dxHat.copy();
        MatrixOps.addColumnsInplace(dLdx, meanDxHat.uminus());
        dLdx.addInplace(-1.0f, MatrixOps.mulColumnsInplace(xHat, meanDxHatXHat));
        MatrixOps.mulColumnsInplace(dLdx, invStd);

        xHat = null;
        invStd = null;
        input = null; // inherited from AbstractLayer

        return dLdx;
    }

    /**
     * Writes gamma and beta to {@code os} in that order.
     *
     * @param os the stream to write into; the caller closes it
     * @throws IOException if writing fails
     */
    void writeParameters(OutputStream os) throws IOException {
        Matrices.serializeF(gamma, os);
        Matrices.serializeF(beta, os);
    }

    /**
     * Reads gamma and beta back in the order {@link #writeParameters(OutputStream)}
     * wrote them.
     *
     * @param is the stream to read from; the caller closes it
     * @throws IOException if reading fails
     */
    void readParameters(InputStream is) throws IOException {
        gamma.setInplace(Matrices.deserializeF(is));
        beta.setInplace(Matrices.deserializeF(is));
    }

    /**
     * Persists gamma and beta if this layer was constructed with storing enabled.
     */
    @Override
    public void storeParameters() {
        if (!storeParameters) {
            return;
        }
        try {
            Files.createDirectories(Paths.get(STORE_DIR));
            try (FileOutputStream fos = new FileOutputStream(STORE_DIR + "ln_" + name)) {
                writeParameters(fos);
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
