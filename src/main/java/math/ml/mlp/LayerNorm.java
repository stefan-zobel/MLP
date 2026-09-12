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

    // Written by every backward and handed straight to the parameter update, which needs
    // them as matrices (features x 1)
    private final MatrixF dGamma;
    private final MatrixF dBeta;

    // One entry per sample, written by a TRAIN forward and read by the backward that
    // follows it. Reused across batches and reallocated only when the batch size changes;
    // backward recomputes the normalized input from these and the cached input.
    private float[] mean;
    private float[] invStd;

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
        dGamma = Matrices.createF(features, 1);
        dBeta = Matrices.createF(features, 1);

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

        int m = input.numColumns();
        float[] mu;
        float[] inv;
        if (mode == NetworkMode.TRAIN) {
            if (mean == null || mean.length != m) {
                mean = new float[m];
                invStd = new float[m];
            }
            mu = mean;
            inv = invStd;
        } else {
            // inference keeps nothing: writing the fields would disturb what a TRAIN
            // forward left behind for the backward that has not run yet
            mu = new float[m];
            inv = new float[m];
        }

        float[] x = input.getArrayUnsafe();
        statistics(x, m, mu, inv);

        MatrixF output = Matrices.sameDimF(input);
        normalize(x, mu, inv, output.getArrayUnsafe(), 0, x.length);
        return output;
    }

    /**
     * Fills the per-sample mean and inverse standard deviation of the
     * {@code features x m} matrix behind {@code x}.
     *
     * <p>Both reductions run over one column at a time, accumulating in {@code double}
     * and scaling by the {@code float} reciprocal of the <em>feature</em> count at the
     * end, because that is what {@code Matrices.rowsAverage} does and the numbers must
     * not move.
     *
     * @param x   the input values, column-major
     * @param m   the number of columns
     * @param mu  receives the mean of each column
     * @param inv receives {@code 1/sqrt(variance + eps)} of each column
     */
    private void statistics(float[] x, int m, float[] mu, float[] inv) {
        float reciprocal = 1.0f / features;
        for (int c = 0, i = 0; c < m; ++c, i += features) {
            double sum = 0.0;
            for (int r = 0; r < features; ++r) {
                sum += x[i + r];
            }
            float columnMean = (float) sum * reciprocal;
            mu[c] = columnMean;

            sum = 0.0;
            for (int r = 0; r < features; ++r) {
                // the square rounds to float before it is accumulated, as hadamard does
                float centered = x[i + r] - columnMean;
                sum += centered * centered;
            }
            inv[c] = (float) (1.0 / Math.sqrt((float) sum * reciprocal + eps));
        }
    }

    /**
     * Writes the normalized, scaled and shifted output for the elements
     * {@code [from, to)} of the column-major backing arrays.
     *
     * <p>Both bounds are column boundaries, that is multiples of {@code features}, which
     * is what makes a range of columns a contiguous range of elements here.
     *
     * @param x    the input values
     * @param mu   the mean of each column
     * @param inv  the inverse standard deviation of each column
     * @param out  where to write {@code gamma * (x - mu) * inv + beta}
     * @param from first element to write, a multiple of {@code features}
     * @param to   one past the last element to write, a multiple of {@code features}
     */
    private void normalize(float[] x, float[] mu, float[] inv, float[] out, int from, int to) {
        float[] g = gamma.getArrayUnsafe();
        float[] b = beta.getArrayUnsafe();
        for (int i = from; i < to; i += features) {
            int c = i / features;
            for (int r = 0; r < features; ++r) {
                out[i + r] = (x[i + r] - mu[c]) * inv[c] * g[r] + b[r];
            }
        }
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
        float[] dy = dLdy.getArrayUnsafe();

        // one pass for all four reductions: two per feature for gamma and beta, two per
        // sample for the two means the input gradient needs
        float[] meanDxHat = new float[m];
        float[] meanDxHatXHat = new float[m];
        reduce(dy, m, meanDxHat, meanDxHatXHat);

        // dL/dx = invStd * (dxHat - mean_i(dxHat) - xHat * mean_i(dxHat * xHat)),
        // both means taken per sample, i.e. over the features of one column
        MatrixF dLdx = Matrices.sameDimF(dLdy);
        inputGradient(dy, meanDxHat, meanDxHatXHat, dLdx.getArrayUnsafe(), 0, dy.length);

        // last of all, because everything above reads the gamma the forward pass used
        gamma.addInplace(-learningRate / m, dGamma);
        beta.addInplace(-learningRate / m, dBeta);

        input = null; // inherited from AbstractLayer
        return dLdx;
    }

    /**
     * Runs the four reductions of the backward pass in one walk over the batch: the
     * gradients of {@code gamma} and {@code beta} into their fields, and the two
     * per-sample means the input gradient needs.
     *
     * <p>Each accumulator sees the same terms in the same order as the
     * {@code Matrices.sumColumns} or {@code Matrices.rowsAverage} call it replaces, in
     * {@code double} and narrowed once. One traversal serves both directions: a per-row
     * sum wants ascending columns and a per-column sum wants ascending rows, and walking
     * the array in order gives each of them exactly that.
     *
     * @param dy            the incoming gradient
     * @param m             the number of columns
     * @param meanDxHat     receives the per-sample mean of {@code dxHat}
     * @param meanDxHatXHat receives the per-sample mean of {@code dxHat * xHat}
     */
    private void reduce(float[] dy, int m, float[] meanDxHat, float[] meanDxHatXHat) {
        float[] x = input.getArrayUnsafe();
        float[] g = gamma.getArrayUnsafe();
        double[] sumGamma = new double[features];
        double[] sumBeta = new double[features];
        float reciprocal = 1.0f / features;

        for (int c = 0, i = 0; c < m; ++c, i += features) {
            double sumDxHat = 0.0;
            double sumDxHatXHat = 0.0;
            for (int r = 0; r < features; ++r) {
                float dl = dy[i + r];
                float dxHat = dl * g[r];
                // the same expression the forward pass used, rather than a cached copy
                float xHat = (x[i + r] - mean[c]) * invStd[c];
                sumGamma[r] += dl * xHat;
                sumBeta[r] += dl;
                sumDxHat += dxHat;
                sumDxHatXHat += dxHat * xHat;
            }
            meanDxHat[c] = (float) sumDxHat * reciprocal;
            meanDxHatXHat[c] = (float) sumDxHatXHat * reciprocal;
        }

        float[] dg = dGamma.getArrayUnsafe();
        float[] db = dBeta.getArrayUnsafe();
        for (int r = 0; r < features; ++r) {
            dg[r] = (float) sumGamma[r];
            db[r] = (float) sumBeta[r];
        }
    }

    /**
     * Writes the input gradient for the elements {@code [from, to)} of the column-major
     * backing arrays. Both bounds are multiples of {@code features}.
     *
     * @param dy            the incoming gradient
     * @param meanDxHat     the per-sample mean of {@code dxHat}
     * @param meanDxHatXHat the per-sample mean of {@code dxHat * xHat}
     * @param out           where to write the result
     * @param from          first element to write
     * @param to            one past the last element to write
     */
    private void inputGradient(float[] dy, float[] meanDxHat, float[] meanDxHatXHat, float[] out,
            int from, int to) {
        float[] x = input.getArrayUnsafe();
        float[] g = gamma.getArrayUnsafe();
        for (int i = from; i < to; i += features) {
            int c = i / features;
            for (int r = 0; r < features; ++r) {
                float dxHat = dy[i + r] * g[r];
                float xHat = (x[i + r] - mean[c]) * invStd[c];
                out[i + r] = (dxHat - meanDxHat[c] - xHat * meanDxHatXHat[c]) * invStd[c];
            }
        }
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
