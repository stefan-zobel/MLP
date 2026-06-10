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

import java.util.concurrent.ThreadLocalRandom;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

/**
 * Reparameterization layer for a Variational Autoencoder (VAE).
 *
 * <h3>Input contract</h3>
 * Expects a {@code (2 * latentDim) × m} matrix whose first {@code latentDim}
 * rows are the posterior <em>mean vectors</em> ? and whose remaining rows are
 * the <em>log-variance vectors</em> log ?².  Such a matrix is naturally
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
 * <h3>Forward pass (TRAIN mode)</h3>
 * <ol>
 *   <li>Splits input into ? (rows 0..latentDim-1) and log ?²
 *       (rows latentDim..2·latentDim-1).</li>
 *   <li>Samples ? ~ N(0, I) element-wise.</li>
 *   <li>Computes ? = exp(log ?² / 2) and returns
 *       <b>z = ? + ? ? ?</b> &nbsp; (shape: {@code latentDim × m}).</li>
 * </ol>
 * ? and ? are cached for use in the backward pass.
 *
 * <h3>Forward pass (INFER mode)</h3>
 * Returns ? directly (deterministic encoding, no sampling).
 *
 * <h3>Backward pass</h3>
 * Given the reconstruction gradient ?L_recon/?z from the downstream decoder,
 * this layer adds the KL-divergence gradient contributions and returns the
 * combined gradient w.r.t. the concatenated input [{@code dL/d? ; dL/d(log ?²)}]
 * of shape {@code (2 * latentDim) × m}:
 * <ul>
 *   <li><b>?L/??</b> = ?L_recon/?z + ? · ?
 *       &nbsp;&nbsp;(KL term: ?KL/?? = ?)</li>
 *   <li><b>?L/?(log ?²)</b> = ?L_recon/?z ? ? ? ? / 2 + ? · (?² ? 1) / 2
 *       &nbsp;&nbsp;(chain rule + KL term: ?KL/?log ?² = (?²?1)/2)</li>
 * </ul>
 * where ? is the KL weight (1.0 for a standard VAE; set &lt; 1 for a ?-VAE
 * to relax the bottleneck).
 *
 * <p>The returned gradient has exactly the shape of the forward input so that
 * {@link ParallelBranches#backward} can split it correctly.
 */
public class VAEReparamLayer extends AbstractLayer {

    private final int latentDim;

    /**
     * Weight ? applied to the KL-divergence gradient.  Use 1.0 for a standard
     * VAE; smaller values (?-VAE) reduce the regularisation pressure and allow
     * a more expressive latent space.
     */
    private final float klWeight;

    // State cached during forward() for use in backward()
    private MatrixF mu;      // ? : latentDim × m
    private MatrixF logVar;  // log ?² : latentDim × m
    private MatrixF epsilon; // ? ~ N(0,I) : latentDim × m
    private MatrixF sigma;   // ? = exp(logVar/2) : latentDim × m

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    /**
     * Creates a {@code ReparamLayer} with KL weight ? = 1.0 (standard VAE).
     *
     * @param latentDim dimensionality of the latent space
     */
    public VAEReparamLayer(int latentDim) {
        this(latentDim, 1.0f);
    }

    /**
     * Creates a {@code ReparamLayer} with an explicit KL weight (?-VAE).
     *
     * @param latentDim dimensionality of the latent space
     * @param klWeight  weight ? applied to the KL-divergence gradient term
     */
    public VAEReparamLayer(int latentDim, float klWeight) {
        this.latentDim = latentDim;
        this.klWeight = klWeight;
    }

    // -------------------------------------------------------------------------
    // Layer contract
    // -------------------------------------------------------------------------

    /**
     * Applies the reparameterization trick:
     * z = ? + ? ? ?, &nbsp; ? ~ N(0, I).
     *
     * <p>In {@code INFER} mode returns ? directly without sampling.
     *
     * @param input (2·latentDim × m) matrix: rows 0..latentDim-1 = ?,
     *              rows latentDim..2·latentDim-1 = log ?²
     * @return z of shape (latentDim × m) in TRAIN mode, ? in INFER mode
     */
    @Override
    public MatrixF forward(MatrixF input) {
        int cols = input.numColumns();
        mu = selectRows(input, 0, latentDim - 1);

        if (mode == NetworkMode.INFER) {
            // Deterministic: use the posterior mean directly.
            return mu;
        }

        logVar = selectRows(input, latentDim, 2 * latentDim - 1);
        sigma   = Matrices.createF(latentDim, cols);
        epsilon = Matrices.createF(latentDim, cols);
        MatrixF z = Matrices.createF(latentDim, cols);

        for (int c = 0; c < cols; c++) {
            for (int r = 0; r < latentDim; r++) {
                float lv  = logVar.getUnsafe(r, c);
                float sig = (float) Math.exp(0.5 * lv);
                float eps = (float) ThreadLocalRandom.current().nextGaussian();
                sigma.setUnsafe(r, c, sig);
                epsilon.setUnsafe(r, c, eps);
                z.setUnsafe(r, c, mu.getUnsafe(r, c) + eps * sig);
            }
        }
        return z;
    }

    /**
     * Computes the combined gradient w.r.t. the concatenated input
     * [? ; log ?²], incorporating both the reconstruction gradient and the
     * KL-divergence gradient.
     *
     * @param dLdz the gradient ?L_recon/?z arriving from the decoder
     *             (latentDim × m)
     * @return [?L/?? ; ?L/?(log ?²)] of shape (2·latentDim × m);
     *         {@code null} in INFER mode
     */
    @Override
    public MatrixF backward(MatrixF dLdz, float unused) {
        if (mode == NetworkMode.INFER) {
            return null;
        }
        int cols = dLdz.numColumns();
        MatrixF dLdMu     = Matrices.createF(latentDim, cols);
        MatrixF dLdLogVar = Matrices.createF(latentDim, cols);

        for (int c = 0; c < cols; c++) {
            for (int r = 0; r < latentDim; r++) {
                float grad   = dLdz.getUnsafe(r, c);
                float sig    = sigma.getUnsafe(r, c);
                float eps    = epsilon.getUnsafe(r, c);
                float muVal  = mu.getUnsafe(r, c);
                float sigSq  = sig * sig; // ?²

                // ?L/?? = ?L_recon/?z  +  ? · ?
                //         (decoder grad)  (KL: ?KL/?? = ?)
                dLdMu.setUnsafe(r, c, grad + klWeight * muVal);

                // ?L/?(log ?²) = ?L_recon/?z · ? · ? / 2  +  ? · (?² ? 1) / 2
                //                (chain rule ?z/?logVar)    (KL: ?KL/?logVar)
                dLdLogVar.setUnsafe(r, c,
                        grad * eps * sig * 0.5f + klWeight * (sigSq - 1.0f) * 0.5f);
            }
        }

        // Release cached state to allow GC.
        mu      = null;
        logVar  = null;
        epsilon = null;
        sigma   = null;

        // Return [?L/?? ; ?L/?(log ?²)] – same row layout as the forward input,
        // so ParallelBranches.backward() can split it without any extra knowledge.
        return stackRows(dLdMu, dLdLogVar);
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /**
     * Returns a new matrix containing rows {@code fromRow} through
     * {@code toRow} (inclusive) of {@code m}.
     */
    private static MatrixF selectRows(MatrixF m, int fromRow, int toRow) {
        int rows = toRow - fromRow + 1;
        int cols = m.numColumns();
        MatrixF result = Matrices.createF(rows, cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                result.setUnsafe(r, c, m.getUnsafe(fromRow + r, c));
            }
        }
        return result;
    }

    /**
     * Vertically stacks {@code top} and {@code bottom} into a single new
     * matrix (top rows first).
     */
    private static MatrixF stackRows(MatrixF top, MatrixF bottom) {
        int cols  = top.numColumns();
        int topR  = top.numRows();
        int botR  = bottom.numRows();
        MatrixF result = Matrices.createF(topR + botR, cols);
        for (int r = 0; r < topR; r++) {
            for (int c = 0; c < cols; c++) {
                result.setUnsafe(r, c, top.getUnsafe(r, c));
            }
        }
        for (int r = 0; r < botR; r++) {
            for (int c = 0; c < cols; c++) {
                result.setUnsafe(topR + r, c, bottom.getUnsafe(r, c));
            }
        }
        return result;
    }
}
