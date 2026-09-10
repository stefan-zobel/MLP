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

import static math.ml.mlp.GradientCheck.assertInputGradient;
import static math.ml.mlp.GradientCheck.input;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import net.jamu.matrix.MatrixF;

class ResidualBranchTest {

    @Test
    void backwardMatchesNumericalInputGradient() {
        // Sigmoid, not Relu: central differences are only valid on a smooth
        // function, and a step of H can flip a Relu across its kink. Relu's own
        // derivative is covered in ActivationTest.
        ResidualBranch layer = new ResidualBranch(new Hidden(6, 6, "rb1"), new BatchNorm(6), new Sigmoid());
        assertInputGradient(layer, input(6, 5, 21L), input(6, 5, 22L), 3e-2);
    }

    @Test
    void backwardMatchesNumericalInputGradientWithoutAnActivation() {
        // the branch above contains a layer that may mutate, this one does not, so
        // the two exercise the copied and the uncopied path through the same code
        ResidualBranch layer = new ResidualBranch(new Hidden(6, 6, "rb2"), new BatchNorm(6));
        assertInputGradient(layer, input(6, 5, 31L), input(6, 5, 32L), 3e-2);
    }

    @Test
    void aMutatingBranchLayerCannotReachTheIdentityPath() {
        ResidualBranch layer = new ResidualBranch(new MutatingProbe(4, 3, 1.0f));
        layer.setMode(NetworkMode.TRAIN);
        MatrixF x = input(4, 3, 33L);
        MatrixF out = layer.forward(x.copy());
        for (int c = 0; c < x.numColumns(); ++c) {
            for (int r = 0; r < x.numRows(); ++r) {
                assertEquals(x.getUnsafe(r, c) + 1.0f, out.getUnsafe(r, c), 1e-5f);
            }
        }

        MatrixF grads = input(4, 3, 34L);
        MatrixF inputGrads = layer.backward(grads.copy());
        for (int c = 0; c < grads.numColumns(); ++c) {
            for (int r = 0; r < grads.numRows(); ++r) {
                assertEquals(grads.getUnsafe(r, c), inputGrads.getUnsafe(r, c), 1e-6f);
            }
        }
    }

    @Test
    void forwardAddsTheIdentityShortcut() {
        // an empty branch makes F(x) = x, so the output must be exactly 2x
        ResidualBranch layer = new ResidualBranch();
        layer.setMode(NetworkMode.TRAIN);
        MatrixF x = input(4, 3, 23L);
        MatrixF out = layer.forward(x.copy());
        for (int c = 0; c < x.numColumns(); ++c) {
            for (int r = 0; r < x.numRows(); ++r) {
                assertEquals(2.0f * x.getUnsafe(r, c), out.getUnsafe(r, c), 1e-5f);
            }
        }
    }

    @Test
    void forwardRejectsABranchThatChangesTheShape() {
        ResidualBranch layer = new ResidualBranch(new Hidden(6, 4, "rb2"));
        layer.setMode(NetworkMode.TRAIN);
        assertThrows(IllegalStateException.class, () -> layer.forward(input(6, 5, 24L)));
    }

    @Test
    void backwardReturnsNullInInferenceMode() {
        ResidualBranch layer = new ResidualBranch(new Hidden(4, 4, "rb3"));
        layer.setMode(NetworkMode.INFER);
        layer.forward(input(4, 3, 25L));
        assertNull(layer.backward(input(4, 3, 26L)));
    }
}
