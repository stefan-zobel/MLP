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
import java.util.List;

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
 * Full analytic gradient through all three statistics. The gradients of {@code gamma}
 * and {@code beta} are left in their {@link Parameter} buffers, averaged over the
 * batch as every other layer leaves them; an {@link Optimizer} applies them.
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

    // Learnable parameters (features x 1); each carries its own gradient buffer
    private final Parameter gamma;     // scale,  initialized to 1
    private final Parameter beta;      // shift,  initialized to 0

    // Running statistics for inference (features x 1)
    private final MatrixF runningMean; // initialized to 0
    private final MatrixF runningVar;  // initialized to 1

    // Written by every TRAIN forward and read by the backward that follows it. All three
    // are features x 1, a shape known at construction, so they are allocated once.
    private final MatrixF batchMean; // per-feature mean
    private final MatrixF batchVar;  // per-feature variance, kept for the running average
    private final MatrixF invStd;    // 1/sqrt(var+eps) per feature

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

        // gamma = 1, runningVar = 1; beta and runningMean stay 0. Neither gamma nor beta
        // is weight decayed, the same rule every framework follows for a scale and shift.
        gamma       = new Parameter("gamma", Matrices.onesF(features, 1), false);
        beta        = new Parameter("beta", Matrices.createF(features, 1), false);
        runningMean = Matrices.createF(features, 1);
        runningVar  = Matrices.onesF(features, 1);

        batchMean   = Matrices.createF(features, 1);
        batchVar    = Matrices.createF(features, 1);
        invStd      = Matrices.createF(features, 1);

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
        Matrices.serializeF(gamma.value(), os);
        Matrices.serializeF(beta.value(), os);
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
        gamma.value().setInplace(Matrices.deserializeF(is));
        beta.value().setInplace(Matrices.deserializeF(is));
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

    /** The scale and the shift; the running statistics are not trained. */
    @Override
    public List<Parameter> parameters() {
        return List.of(gamma, beta);
    }

    @Override
    public MatrixF forward(MatrixF input) {
        if (input.numRows() != features) {
            throw new IllegalArgumentException(
                    "expected " + features + " features, got " + input.numRows());
        }
        super.forward(input); // caches this.input when mode == TRAIN

        if (mode == NetworkMode.INFER) {
            return applyAffine(input, runningMean, runningVar);
        }

        // --- TRAIN ---

        // 1. and 2. Batch mean, variance and inverse standard deviation per feature
        batchStatistics(input.getArrayUnsafe(), input.numColumns());

        // 3. Normalize and apply gamma and beta. Nothing of batch size is cached for
        //    backward: it recomputes the normalized value from the input it already has.
        MatrixF output = Matrices.sameDimF(input);
        normalize(input.getArrayUnsafe(), output.getArrayUnsafe(), 0, output.getArrayUnsafe().length);

        // 4. Update running statistics (exponential moving average)
        runningMean.scaleInplace(1.0f - momentum).addInplace(momentum, batchMean);
        runningVar.scaleInplace(1.0f - momentum).addInplace(momentum, batchVar);

        return output;
    }

    /**
     * Fills {@link #batchMean}, {@link #batchVar} and {@link #invStd} from the
     * {@code features x m} matrix behind {@code x}.
     *
     * <p>Two passes, both accumulating in {@code double} and scaling by the {@code float}
     * reciprocal of {@code m} at the end, because that is what {@code Matrices.colsAverage}
     * does and the numbers must not move. Walking columns outermost gives the same
     * accumulator the same terms in the same order, but sequentially in memory rather
     * than with a stride of {@code features}.
     */
    private void batchStatistics(float[] x, int m) {
        float[] mean = batchMean.getArrayUnsafe();
        float[] var = batchVar.getArrayUnsafe();
        float[] inv = invStd.getArrayUnsafe();
        double[] sum = new double[features];

        for (int i = 0; i < x.length; i += features) {
            for (int r = 0; r < features; ++r) {
                sum[r] += x[i + r];
            }
        }
        float reciprocal = 1.0f / m;
        for (int r = 0; r < features; ++r) {
            mean[r] = (float) sum[r] * reciprocal;
            sum[r] = 0.0;
        }

        for (int i = 0; i < x.length; i += features) {
            for (int r = 0; r < features; ++r) {
                // the square rounds to float before it is accumulated, as hadamard does
                float centered = x[i + r] - mean[r];
                sum[r] += centered * centered;
            }
        }
        for (int r = 0; r < features; ++r) {
            var[r] = (float) sum[r] * reciprocal;
            inv[r] = (float) (1.0 / Math.sqrt(var[r] + eps));
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
     * @param out  where to write {@code gamma * (x - mean) * invStd + beta}
     * @param from first element to write, a multiple of {@code features}
     * @param to   one past the last element to write, a multiple of {@code features}
     */
    private void normalize(float[] x, float[] out, int from, int to) {
        float[] mean = batchMean.getArrayUnsafe();
        float[] inv = invStd.getArrayUnsafe();
        float[] g = gamma.value().getArrayUnsafe();
        float[] b = beta.value().getArrayUnsafe();
        for (int i = from; i < to; i += features) {
            for (int r = 0; r < features; ++r) {
                out[i + r] = (x[i + r] - mean[r]) * inv[r] * g[r] + b[r];
            }
        }
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
    public MatrixF backward(MatrixF dLdy) {
        if (mode == NetworkMode.INFER) {
            return null;
        }
        int m = dLdy.numColumns();

        // one pass for all four reductions; dxHat and (x - mean) are one multiply and one
        // subtract each, which is cheaper than the two batch-sized matrices they were
        float[] dVar = new float[features];
        float[] dMean = new float[features];
        reduce(dLdy.getArrayUnsafe(), dVar, dMean);

        // --- dL/dx = dxHat*invStd + dVar*2*(x-mean)/m + dMean/m ---
        MatrixF dLdx = Matrices.sameDimF(dLdy);
        inputGradient(dLdy.getArrayUnsafe(), dVar, dMean, m, dLdx.getArrayUnsafe(), 0,
                dLdx.getArrayUnsafe().length);

        // The reduction left the raw sum over the batch in the two gradient buffers;
        // averaging here is what makes every parameter gradient in this framework a mean.
        // Nothing is applied to gamma or beta: the Optimizer does that after the whole
        // backward pass, which is also why gamma still holds what the forward pass used.
        gamma.grad().scaleInplace(1.0f / m);
        beta.grad().scaleInplace(1.0f / m);

        // The only cached matrix left is the caller's input; batchMean, batchVar and
        // invStd are features x 1 and are simply overwritten by the next forward.
        input = null; // inherited from AbstractLayer

        return dLdx;
    }

    /**
     * Runs the four per-feature reductions of the backward pass in one walk over the
     * batch: the gradients of {@code gamma} and {@code beta} into their fields, and the
     * gradients of the variance and the mean into {@code dVar} and {@code dMean}.
     *
     * <p>Each accumulator sees the same terms in the same order as the
     * {@code Matrices.sumColumns} call it replaces, in {@code double} and narrowed once.
     *
     * @param dy    the incoming gradient
     * @param dVar  receives {@code dL/dvar}, one entry per feature
     * @param dMean receives {@code dL/dmean} already divided by the batch size
     */
    private void reduce(float[] dy, float[] dVar, float[] dMean) {
        float[] x = input.getArrayUnsafe();
        float[] mean = batchMean.getArrayUnsafe();
        float[] inv = invStd.getArrayUnsafe();
        float[] g = gamma.value().getArrayUnsafe();
        double[] sumGamma = new double[features];
        double[] sumBeta = new double[features];
        double[] sumVar = new double[features];
        double[] sumMean = new double[features];

        for (int i = 0; i < dy.length; i += features) {
            for (int r = 0; r < features; ++r) {
                float dl = dy[i + r];
                float dxHat = dl * g[r];
                // the same expression the forward pass used, rather than a cached copy
                float centered = x[i + r] - mean[r];
                sumGamma[r] += dl * (centered * inv[r]);
                sumBeta[r] += dl;
                sumVar[r] += dxHat * centered;
                sumMean[r] += dxHat;
            }
        }

        float[] dg = gamma.grad().getArrayUnsafe();
        float[] db = beta.grad().getArrayUnsafe();
        float reciprocal = 1.0f / (dy.length / features);
        for (int r = 0; r < features; ++r) {
            dg[r] = (float) sumGamma[r];
            db[r] = (float) sumBeta[r];
            // the three invStd factors and the halving were three separate roundings
            // before and have to stay that way
            dVar[r] = (float) sumVar[r] * inv[r] * inv[r] * inv[r] * -0.5f;
            dMean[r] = (float) sumMean[r] * inv[r] * -1.0f * reciprocal;
        }
    }

    /**
     * Writes the input gradient for the elements {@code [from, to)} of the column-major
     * backing arrays. Both bounds are multiples of {@code features}.
     *
     * @param dy    the incoming gradient
     * @param dVar  {@code dL/dvar} per feature
     * @param dMean {@code dL/dmean} per feature, already divided by {@code m}
     * @param m     the batch size
     * @param out   where to write the result
     * @param from  first element to write
     * @param to    one past the last element to write
     */
    private void inputGradient(float[] dy, float[] dVar, float[] dMean, int m, float[] out, int from,
            int to) {
        float[] x = input.getArrayUnsafe();
        float[] mean = batchMean.getArrayUnsafe();
        float[] inv = invStd.getArrayUnsafe();
        float[] g = gamma.value().getArrayUnsafe();
        float twoOverM = 2.0f / m;
        for (int i = from; i < to; i += features) {
            for (int r = 0; r < features; ++r) {
                float dxHat = dy[i + r] * g[r];
                out[i + r] = dxHat * inv[r] + (x[i + r] - mean[r]) * twoOverM * dVar[r] + dMean[r];
            }
        }
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /**
     * Applies the affine transform {@code y = gamma*(x-mean)/sqrt(var+eps) + beta}
     * using the supplied statistics (used for inference with running stats).
     */
    private MatrixF applyAffine(MatrixF x, MatrixF mean, MatrixF var) {
        // a local rather than the invStd field: inference must not disturb the state a
        // TRAIN forward left behind for its backward
        float[] inv = new float[features];
        float[] v = var.getArrayUnsafe();
        for (int r = 0; r < features; ++r) {
            inv[r] = (float) (1.0 / Math.sqrt(v[r] + eps));
        }

        MatrixF output = Matrices.sameDimF(x);
        float[] in = x.getArrayUnsafe();
        float[] out = output.getArrayUnsafe();
        float[] mu = mean.getArrayUnsafe();
        float[] g = gamma.value().getArrayUnsafe();
        float[] b = beta.value().getArrayUnsafe();
        for (int i = 0; i < in.length; i += features) {
            for (int r = 0; r < features; ++r) {
                out[i + r] = (in[i + r] - mu[r]) * inv[r] * g[r] + b[r];
            }
        }
        return output;
    }
}
