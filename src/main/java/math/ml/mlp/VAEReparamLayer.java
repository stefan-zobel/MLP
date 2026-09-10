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

import java.util.SplittableRandom;
import java.util.concurrent.ThreadLocalRandom;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

/**
 * Reparameterization layer for a Variational Autoencoder (VAE).
 *
 * <h2>Input contract</h2>
 * Expects a {@code (2 * latentDim) x m} matrix whose first {@code latentDim}
 * rows are the posterior <em>mean vectors</em> &mu; and whose remaining rows are
 * the <em>log-variance vectors</em> log &sigma;&sup2;.  Such a matrix is naturally
 * produced by a {@link ParallelBranches} layer with two heads of equal output
 * size:
 * <pre>{@code
 * net.add(new ParallelBranches(
 *     List.of(new Hidden(hiddenDim, latentDim, "mu")),
 *     List.of(new Hidden(hiddenDim, latentDim, "logvar"))
 * ));
 * net.add(new ReparamLayer(latentDim));
 * }</pre>
 *
 * <h2>Forward pass (TRAIN mode)</h2>
 * <ol>
 *   <li>Splits input into &mu; (rows 0..latentDim-1) and log &sigma;&sup2;
 *       (rows latentDim..2&middot;latentDim-1).</li>
 *   <li>Samples &epsilon; ~ N(0, I) element-wise.</li>
 *   <li>Computes &sigma; = exp(log &sigma;&sup2; / 2) and returns
 *       <b>z = &mu; + &sigma; &#8857; &epsilon;</b> &nbsp; (shape: {@code latentDim x m}).</li>
 * </ol>
 * &epsilon; and &sigma; are cached for use in the backward pass.
 *
 * <h2>Forward pass (INFER mode)</h2>
 * Returns &mu; directly (deterministic encoding, no sampling).
 *
 * <h2>Backward pass</h2>
 * Given the reconstruction gradient &part;L_recon/&part;z from the downstream decoder,
 * this layer adds the KL-divergence gradient contributions and returns the
 * combined gradient w.r.t. the concatenated input [{@code dL/dmu ; dL/d(log sigma^2)}]
 * of shape {@code (2 * latentDim) x m}:
 * <ul>
 *   <li><b>&part;L/&part;&mu;</b> = &part;L_recon/&part;z + &lambda; &middot; &mu;
 *       &nbsp;&nbsp;(KL term: &part;KL/&part;&mu; = &mu;)</li>
 *   <li><b>&part;L/&part;(log &sigma;&sup2;)</b> = &part;L_recon/&part;z &#8857; &epsilon; &#8857; &sigma; / 2 + &lambda; &middot; (&sigma;&sup2; &minus; 1) / 2
 *       &nbsp;&nbsp;(chain rule + KL term: &part;KL/&part;log &sigma;&sup2; = (&sigma;&sup2;&minus;1)/2)</li>
 * </ul>
 * where &lambda; is the KL weight (1.0 for a standard VAE; set &lt; 1 for a &beta;-VAE
 * to relax the bottleneck).
 *
 * <p>The returned gradient has exactly the shape of the forward input so that
 * {@link ParallelBranches#backward} can split it correctly.
 */
public class VAEReparamLayer extends AbstractLayer {

    private final int latentDim;

    /**
     * Weight &lambda; applied to the KL-divergence gradient.  Use 1.0 for a standard
     * VAE; smaller values (&beta;-VAE) reduce the regularization pressure and allow
     * a more expressive latent space.
     */
    private final float klWeight;

    // State cached during forward() for use in backward()
    private MatrixF mu;      // mu : latentDim x m
    private MatrixF logVar;  // log sigma^2 : latentDim x m
    private MatrixF epsilon; // eps ~ N(0,I) : latentDim x m
    private MatrixF sigma;   // sigma = exp(logVar/2) : latentDim x m

    /** Owned by this layer; epsilon is drawn from it once per forward pass. */
    private final SplittableRandom rng;

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    /**
     * Creates a {@code VAEReparamLayer} with KL weight &lambda; = 1.0 (standard VAE).
     *
     * @param latentDim dimensionality of the latent space
     */
    public VAEReparamLayer(int latentDim) {
        this(latentDim, 1.0f);
    }

    /**
     * Creates a {@code VAEReparamLayer} with an explicit KL weight (&beta;-VAE).
     *
     * @param latentDim dimensionality of the latent space
     * @param klWeight  weight &lambda; applied to the KL-divergence gradient term
     */
    public VAEReparamLayer(int latentDim, float klWeight) {
        this(latentDim, klWeight, ThreadLocalRandom.current().nextLong());
    }

    /**
     * Creates a {@code VAEReparamLayer} whose sampling is reproducible. A fresh
     * &epsilon; is drawn per forward pass, so {@code seed} fixes the whole sequence.
     *
     * @param latentDim dimensionality of the latent space
     * @param klWeight  weight &lambda; applied to the KL-divergence gradient term
     * @param seed      seed for the &epsilon; draws; INFER mode draws nothing
     */
    public VAEReparamLayer(int latentDim, float klWeight, long seed) {
        this.latentDim = latentDim;
        this.klWeight = klWeight;
        this.rng = new SplittableRandom(seed);
    }

    // -------------------------------------------------------------------------
    // Layer contract
    // -------------------------------------------------------------------------

    /**
     * Applies the reparameterization trick:
     * z = &mu; + &sigma; &#8857; &epsilon;, &nbsp; &epsilon; ~ N(0, I).
     *
     * <p>In {@code INFER} mode returns &mu; directly without sampling.
     *
     * @param input (2&middot;latentDim &times; m) matrix: rows 0..latentDim-1 = &mu;,
     *              rows latentDim..2&middot;latentDim-1 = log &sigma;&sup2;
     * @return z of shape (latentDim &times; m) in TRAIN mode, &mu; in INFER mode
     */
    @Override
    public MatrixF forward(MatrixF input) {
        int cols = input.numColumns();
        mu = input.selectSubmatrix(0, 0, latentDim - 1, cols - 1);

        if (mode == NetworkMode.INFER) {
            // Deterministic: use the posterior mean directly.
            return mu;
        }

        logVar = input.selectSubmatrix(latentDim, 0, 2 * latentDim - 1, cols - 1);
        sigma = logVar.map(lv -> (float) Math.exp(0.5 * lv));
        epsilon = Matrices.randomNormalF(latentDim, cols, rng.nextLong());
        return mu.plus(epsilon.hadamard(sigma));
    }

    /**
     * Computes the combined gradient w.r.t. the concatenated input
     * [&mu; ; log &sigma;&sup2;], incorporating both the reconstruction gradient and the
     * KL-divergence gradient.
     *
     * @param dLdz the gradient &part;L_recon/&part;z arriving from the decoder
     *             (latentDim &times; m)
     * @return [&part;L/&part;&mu; ; &part;L/&part;(log &sigma;&sup2;)] of shape (2&middot;latentDim &times; m);
     *         {@code null} in INFER mode
     */
    @Override
    public MatrixF backward(MatrixF dLdz, float unused) {
        if (mode == NetworkMode.INFER) {
            return null;
        }
        int cols = dLdz.numColumns();
        float kl = klWeight;

        // dL/dmu = dL_recon/dz + lambda * mu
        MatrixF dLdMu = mu.scale(kl, Matrices.sameDimF(mu)).addInplace(1.0f, dLdz);

        // dL/d(log sigma^2) = dL_recon/dz * eps * sigma / 2 + lambda * (sigma^2 - 1) / 2
        MatrixF dLdLogVar = dLdz.hadamard(epsilon).hadamard(sigma).scaleInplace(0.5f)
                .addInplace(1.0f, sigma.hadamard(sigma).mapInplace(sq -> 0.5f * kl * (sq - 1.0f)));

        // Release cached state to allow GC.
        mu      = null;
        logVar  = null;
        epsilon = null;
        sigma   = null;

        // Return [dL/dmu ; dL/d(log sigma^2)] - same row layout as the forward input,
        // so ParallelBranches.backward() can split it without any extra knowledge.
        return Matrices.createF(2 * latentDim, cols)
                .setSubmatrixInplace(0, 0, dLdMu, 0, 0, latentDim - 1, cols - 1)
                .setSubmatrixInplace(latentDim, 0, dLdLogVar, 0, 0, latentDim - 1, cols - 1);
    }
}
