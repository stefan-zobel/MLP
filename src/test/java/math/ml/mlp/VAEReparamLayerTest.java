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

import static math.ml.mlp.GradientCheck.field;
import static math.ml.mlp.GradientCheck.input;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

/**
 * Epsilon is redrawn on every forward pass, so most of these tests assert the
 * layer's identities against its own cached state rather than against finite
 * differences.
 */
class VAEReparamLayerTest {

    private static final int LATENT = 5;
    private static final int BATCH = 4;

    @Test
    void forwardAppliesTheReparameterizationTrick() throws Exception {
        VAEReparamLayer layer = new VAEReparamLayer(LATENT);
        layer.setMode(NetworkMode.TRAIN);
        MatrixF in = input(2 * LATENT, BATCH, 31L);
        MatrixF z = layer.forward(in);

        MatrixF mu = field(layer, "mu");
        MatrixF logVar = field(layer, "logVar");
        MatrixF eps = field(layer, "epsilon");
        MatrixF sigma = field(layer, "sigma");

        for (int c = 0; c < BATCH; ++c) {
            for (int r = 0; r < LATENT; ++r) {
                assertEquals(in.getUnsafe(r, c), mu.getUnsafe(r, c), 1e-6f, "mu must be the top rows");
                assertEquals(Math.exp(0.5 * logVar.getUnsafe(r, c)), sigma.getUnsafe(r, c), 1e-5,
                        "sigma = exp(logVar/2)");
                assertEquals(mu.getUnsafe(r, c) + eps.getUnsafe(r, c) * sigma.getUnsafe(r, c), z.getUnsafe(r, c),
                        1e-5f, "z = mu + eps * sigma");
            }
        }
    }

    @Test
    void backwardPropagatesTheReconstructionGradient() throws Exception {
        VAEReparamLayer layer = new VAEReparamLayer(LATENT, 0.0f);
        layer.setMode(NetworkMode.TRAIN);
        layer.forward(input(2 * LATENT, BATCH, 32L));

        // capture before backward(), which releases the cached state
        MatrixF eps = GradientCheck.<MatrixF>field(layer, "epsilon").copy();
        MatrixF sigma = GradientCheck.<MatrixF>field(layer, "sigma").copy();

        MatrixF dLdz = input(LATENT, BATCH, 33L);
        MatrixF g = layer.backward(dLdz, 0.0f);

        for (int c = 0; c < BATCH; ++c) {
            for (int r = 0; r < LATENT; ++r) {
                assertEquals(dLdz.getUnsafe(r, c), g.getUnsafe(r, c), 1e-6f, "dz/dmu is 1");
                double expected = dLdz.getUnsafe(r, c) * eps.getUnsafe(r, c) * sigma.getUnsafe(r, c) * 0.5;
                assertEquals(expected, g.getUnsafe(LATENT + r, c), 1e-5, "dz/dlogVar = eps * sigma / 2");
            }
        }
    }

    @Test
    void backwardAddsTheClosedFormKlGradient() {
        VAEReparamLayer layer = new VAEReparamLayer(LATENT, 1.0f);
        layer.setMode(NetworkMode.TRAIN);
        MatrixF in = input(2 * LATENT, BATCH, 34L);
        layer.forward(in);

        // zero reconstruction gradient isolates the KL term
        MatrixF g = layer.backward(Matrices.createF(LATENT, BATCH), 0.0f);

        for (int c = 0; c < BATCH; ++c) {
            for (int r = 0; r < LATENT; ++r) {
                double mu = in.getUnsafe(r, c);
                double logVar = in.getUnsafe(LATENT + r, c);
                // KL = 0.5 * (mu^2 + sigma^2 - 1 - logVar)
                assertEquals(mu, g.getUnsafe(r, c), 1e-5, "dKL/dmu = mu");
                assertEquals(0.5 * (Math.exp(logVar) - 1.0), g.getUnsafe(LATENT + r, c), 1e-5,
                        "dKL/dlogVar = (sigma^2 - 1) / 2");
            }
        }
    }

    @Test
    void backwardReturnsTheShapeParallelBranchesCanSplit() {
        VAEReparamLayer layer = new VAEReparamLayer(LATENT);
        layer.setMode(NetworkMode.TRAIN);
        layer.forward(input(2 * LATENT, BATCH, 35L));
        MatrixF g = layer.backward(input(LATENT, BATCH, 36L), 0.0f);
        assertEquals(2 * LATENT, g.numRows());
        assertEquals(BATCH, g.numColumns());
    }

    @Test
    void inferenceReturnsTheMeanWithoutSampling() {
        VAEReparamLayer layer = new VAEReparamLayer(LATENT);
        layer.setMode(NetworkMode.INFER);
        MatrixF in = input(2 * LATENT, BATCH, 37L);

        MatrixF first = layer.forward(in);
        MatrixF second = layer.forward(in);
        assertEquals(LATENT, first.numRows());
        for (int c = 0; c < BATCH; ++c) {
            for (int r = 0; r < LATENT; ++r) {
                assertEquals(in.getUnsafe(r, c), first.getUnsafe(r, c), 1e-6f);
                assertEquals(first.getUnsafe(r, c), second.getUnsafe(r, c), 1e-6f, "inference must be deterministic");
            }
        }
    }

    @Test
    void samplingActuallyVariesBetweenForwardPasses() {
        VAEReparamLayer layer = new VAEReparamLayer(LATENT);
        layer.setMode(NetworkMode.TRAIN);
        MatrixF in = input(2 * LATENT, BATCH, 38L);
        MatrixF first = layer.forward(in);
        MatrixF second = layer.forward(in);

        boolean differs = false;
        for (int c = 0; c < BATCH && !differs; ++c) {
            for (int r = 0; r < LATENT && !differs; ++r) {
                differs = first.getUnsafe(r, c) != second.getUnsafe(r, c);
            }
        }
        assertTrue(differs, "training mode must resample epsilon");
    }

    @Test
    void backwardReturnsNullInInferenceMode() {
        VAEReparamLayer layer = new VAEReparamLayer(LATENT);
        layer.setMode(NetworkMode.INFER);
        layer.forward(input(2 * LATENT, BATCH, 39L));
        assertNull(layer.backward(input(LATENT, BATCH, 40L), 0.1f));
    }

    @Test
    void theSameSeedProducesTheSameSequenceOfSamples() {
        VAEReparamLayer a = new VAEReparamLayer(LATENT, 1.0f, 9L);
        VAEReparamLayer b = new VAEReparamLayer(LATENT, 1.0f, 9L);
        a.setMode(NetworkMode.TRAIN);
        b.setMode(NetworkMode.TRAIN);
        MatrixF in = input(2 * LATENT, BATCH, 41L);

        MatrixF previous = null;
        for (int pass = 0; pass < 4; ++pass) {
            MatrixF fromA = a.forward(in.copy());
            MatrixF fromB = b.forward(in.copy());
            assertFalse(differs(fromA, fromB), "the two layers diverged in pass " + pass);
            if (previous != null) {
                // the seed must fix the sequence, not repeat one epsilon forever
                assertTrue(differs(previous, fromA), "pass " + pass + " repeated the previous epsilon");
            }
            previous = fromA.copy();
        }
    }

    @Test
    void aDifferentSeedProducesDifferentSamples() {
        VAEReparamLayer a = new VAEReparamLayer(LATENT, 1.0f, 9L);
        VAEReparamLayer b = new VAEReparamLayer(LATENT, 1.0f, 10L);
        a.setMode(NetworkMode.TRAIN);
        b.setMode(NetworkMode.TRAIN);
        MatrixF in = input(2 * LATENT, BATCH, 42L);
        assertTrue(differs(a.forward(in.copy()), b.forward(in.copy())));
    }

    private static boolean differs(MatrixF a, MatrixF b) {
        for (int c = 0; c < a.numColumns(); ++c) {
            for (int r = 0; r < a.numRows(); ++r) {
                if (a.getUnsafe(r, c) != b.getUnsafe(r, c)) {
                    return true;
                }
            }
        }
        return false;
    }
}
