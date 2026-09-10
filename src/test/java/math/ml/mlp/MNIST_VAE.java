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

import java.security.SecureRandom;
import java.util.List;
import java.util.SplittableRandom;

import math.cern.Arithmetic;
import math.ml.loader.MNIST;
import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;
import net.jamu.matrix.Statistics;

/**
 * A Variational Autoencoder (VAE) trained on MNIST.
 *
 * <h2>Architecture overview</h2>
 * <pre>
 *  Input (784)
 *      &darr;
 *  Hidden(784&rarr;256) + LayerNorm + ReLU   &larr; shared encoder
 *      &darr;
 *  Hidden(256&rarr;128) + LayerNorm + ReLU   &larr; shared encoder
 *      &darr;
 *  +-- ParallelBranches -----------------------------------+
 *  |  Branch 0: Hidden(128&rarr;LATENT)  &rarr; &mu;      (LATENT &times; m) |
 *  |  Branch 1: Hidden(128&rarr;LATENT)  &rarr; log &sigma;&sup2;  (LATENT &times; m) |
 *  +-------------------------------------------------------+
 *      &darr;  (2&middot;LATENT &times; m, rows [0..LATENT-1]=&mu;, [LATENT..2&middot;LATENT-1]=log &sigma;&sup2;)
 *  VAEReparamLayer(LATENT)          &larr; z = &mu; + &sigma;&#8857;&epsilon;, adds KL gradient in bwd
 *      &darr;  (LATENT &times; m)
 *  Hidden(LATENT&rarr;128) + LayerNorm + ReLU &larr; decoder
 *      &darr;
 *  Hidden(128&rarr;256) + LayerNorm + ReLU   &larr; decoder
 *      &darr;
 *  Hidden(256&rarr;784)                 &larr; logits, no activation here
 *      &darr;
 *  SigmoidBCELoss                   &larr; fused sigmoid + reconstruction loss
 * </pre>
 *
 * <h2>No special handling in AbstractNetwork.train() necessary</h2>
 * The outer training loop in {@link AbstractNetwork#train} sees a strictly
 * sequential list of layers and is completely unaware of the internal
 * branching structure inside {@link ParallelBranches}. The split/merge
 * topology is handled transparently:
 * <ul>
 *   <li>Forward: {@code ParallelBranches.forward(x)} fans out {@code x} into
 *       both heads and vertically stacks their outputs.</li>
 *   <li>Backward: {@code ParallelBranches.backward(grad)} splits the incoming
 *       gradient, routes each slice through its branch in reverse, and sums
 *       the input-side gradients (chain rule).</li>
 *   <li>The KL-divergence gradient is injected by
 *       {@code VAEReparamLayer.backward()} &ndash; no separate loss term needed.</li>
 * </ul>
 */
public class MNIST_VAE extends AbstractNetwork {

    // -----------------------------------------------------------------------
    // Hyper-parameters
    // -----------------------------------------------------------------------
    private static final int LATENT_DIM    = 20;
    private static final int BATCH_SIZE    = 128;
    private static final int NUM_EPOCHS    = 50;
    private static final float LOWER       = 0.0f;
    private static final float UPPER       = 1.0f;

    // -----------------------------------------------------------------------
    // Dataset (use only the original 60 000 training images)
    // -----------------------------------------------------------------------
    private static final MatrixF IMAGES =
            Statistics.rescaleInplace(MNIST.getTrainingSetImages(), LOWER, UPPER);
    // Reconstruction target = input image itself
//    private static final MatrixF TARGETS = IMAGES;

    private static final MatrixF TEST_IMAGES =
            Statistics.rescaleInplace(MNIST.getTestSetImages(), LOWER, UPPER);

    private static final int INPUT_DIM            = IMAGES.numRows();   // 784
    private static final int NUM_BATCHES_PER_EPOCH = IMAGES.numColumns() / BATCH_SIZE;

    // -----------------------------------------------------------------------
    // Training state
    // -----------------------------------------------------------------------
    private static int    epoch          = 0;
    private static double epochLossSum   = 0.0;
    private static int    batchesInEpoch = 0;

    // -----------------------------------------------------------------------
    // AbstractNetwork contract
    // -----------------------------------------------------------------------

    @Override
    public void onLossComputationCompleted(MatrixF losses) {
        epochLossSum += Matrices.colsAverage(losses).toScalar();
        ++batchesInEpoch;
    }

    // -----------------------------------------------------------------------
    // Entry point
    // -----------------------------------------------------------------------

    public static void main(String[] args) {

        // pass this seed back as the first argument to repeat a run exactly
        long baseSeed = args.length > 0 ? Long.parseLong(args[0]) : new SecureRandom().nextLong();
        System.out.println("seed: " + baseSeed);
        SplittableRandom seeds = new SplittableRandom(baseSeed);

        MNIST_VAE net = new MNIST_VAE();

        // --- Loss ---------------------------------------------------------
        SigmoidBCELoss bce = new SigmoidBCELoss();
        bce.registerLossCallback(net::onLossComputationCompleted);

        // --- Encoder ------------------------------------------------------
        net.add(new Hidden(INPUT_DIM, 256, "enc1", seeds.nextLong()));
        net.add(new LayerNorm(256));
        net.add(new Relu());
        net.add(new Hidden(256, 128, "enc2", seeds.nextLong()));
        net.add(new LayerNorm(128));
        net.add(new Relu());

        // --- Split: mu and log sigma^2 heads ------------------------------
        net.add(new ParallelBranches(
                List.of(new Hidden(128, LATENT_DIM, "mu", seeds.nextLong())),     // rows [0..LATENT_DIM-1]
                List.of(new Hidden(128, LATENT_DIM, "logvar", seeds.nextLong()))  // rows [LATENT_DIM..2*LATENT_DIM-1]
        ));

        // --- Reparameterization + KL gradient (lambda = 1.0) -------------
        net.add(new VAEReparamLayer(LATENT_DIM, 1.0f, seeds.nextLong()));

        // --- Decoder ------------------------------------------------------
        net.add(new Hidden(LATENT_DIM, 128, "dec1", seeds.nextLong()));
        net.add(new LayerNorm(128));
        net.add(new Relu());
        net.add(new Hidden(128, 256, "dec2", seeds.nextLong()));
        net.add(new LayerNorm(256));
        net.add(new Relu());
        net.add(new Hidden(256, INPUT_DIM, "dec3", seeds.nextLong()));

        // --- Reconstruction loss (targets = inputs) -----------------------
        net.add(bce);

        // -----------------------------------------------------------------------
        // Training loop
        // -----------------------------------------------------------------------
        // 0.010 rather than 0.001: measured over 6 epochs, mean per-pixel BCE on 2000
        // test images drops from about 0.178 to 0.138
        net.optimizer(new Sgd(0.010f));

        long seed = seeds.nextLong();
        Statistics.shuffleColumnsInplace(IMAGES, seed);
        // TARGETS == IMAGES, so it is already shuffled in sync.

        for (int epochIdx = 0; epochIdx < NUM_EPOCHS; ++epochIdx) {
            for (int b = 0; b < NUM_BATCHES_PER_EPOCH; ++b) {
                int startCol = b * BATCH_SIZE;
                MatrixF input = IMAGES.selectConsecutiveColumns(startCol, startCol + BATCH_SIZE - 1);
                // an autoencoder reconstructs its own input: the batch is its own target
                net.train(input, input);
            }

            double avgLoss = Arithmetic.round(epochLossSum / batchesInEpoch, 6);
            System.out.println("epoch " + epoch + "  avg. BCE loss: " + avgLoss);
            epochLossSum   = 0.0;
            batchesInEpoch = 0;
            ++epoch;

            // reshuffle between epochs
            seed = seeds.nextLong();
            Statistics.shuffleColumnsInplace(IMAGES, seed);
        }

        // -----------------------------------------------------------------------
        // Quick reconstruction check on first 10 test images
        // -----------------------------------------------------------------------
        System.out.println("\nReconstruction check (first 10 test images):");
        MatrixF sample    = TEST_IMAGES.selectConsecutiveColumns(0, 9);
        MatrixF recon     = net.infer(sample);
        float   reconLoss = averageBCE(sample, recon);
        System.out.printf("avg. reconstruction BCE loss: %.6f%n", reconLoss);
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    private static float averageBCE(MatrixF target, MatrixF pred) {
        int rows = target.numRows();
        int cols = target.numColumns();
        double sum = 0.0;
        for (int c = 0; c < cols; c++) {
            for (int r = 0; r < rows; r++) {
                float rawP = pred.getUnsafe(r, c);
                // Clamp p and (1-p) independently from the raw prediction.
                // Using 1.0 - clamp(1-p) would cause log(0) when rawP ~= 0,
                // because 1.0f - rawP rounds to 1.0f (float underflow) and
                // clamp(1.0f) = 1.0f, so 1.0 - 1.0f = 0.0 exactly -> log(0) = -Inf.
                float p = clamp(rawP);
                float q = clamp(1.0f - rawP);   // 1-p clamped from the raw value
                float t = target.getUnsafe(r, c);
                sum -= t * Math.log(p) + (1.0 - t) * Math.log(q);
            }
        }
        return (float) (sum / (rows * cols));
    }

    private static float clamp(float x) {
        return Math.max(x, Float.MIN_NORMAL);
    }
}
