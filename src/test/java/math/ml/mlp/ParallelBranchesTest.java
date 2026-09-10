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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

import net.jamu.matrix.MatrixF;

class ParallelBranchesTest {

    @Test
    void backwardMatchesNumericalInputGradient() {
        // Sigmoid rather than Relu, so the central differences stay on a smooth
        // function; see the note in ResidualBranchTest.
        ParallelBranches layer = new ParallelBranches(
                List.of(new Hidden(6, 4, "p1"), new Sigmoid()),
                List.of(new Hidden(6, 4, "p2")));
        assertInputGradient(layer, input(6, 5, 11L), input(8, 5, 12L), 3e-2);
    }

    @Test
    void backwardMatchesNumericalInputGradientWithoutAnActivation() {
        // branch 0 above contains a layer that may mutate, neither branch here does,
        // so the two exercise the copied and the uncopied path through the same code
        ParallelBranches layer = new ParallelBranches(
                List.of(new Hidden(6, 4, "p3")),
                List.of(new Hidden(6, 4, "p4")));
        assertInputGradient(layer, input(6, 5, 31L), input(8, 5, 32L), 3e-2);
    }

    @Test
    void aMutatingBranchLayerCannotReachTheOtherBranch() {
        ParallelBranches layer = new ParallelBranches(
                List.of(new MutatingProbe(4, 3, 1.0f)),
                List.of(new Hidden(4, 4, "p5")));
        layer.setMode(NetworkMode.TRAIN);
        MatrixF out = layer.forward(input(4, 3, 33L));
        for (int c = 0; c < out.numColumns(); ++c) {
            for (int r = 0; r < out.numRows(); ++r) {
                assertTrue(Float.isFinite(out.getUnsafe(r, c)), "NaN leaked into row " + r);
            }
        }
    }

    @Test
    void forwardStacksBranchOutputsInBranchOrder() {
        ParallelBranches layer = new ParallelBranches(
                List.of(new Hidden(6, 4, "s1")),
                List.of(new Hidden(6, 3, "s2")));
        layer.setMode(NetworkMode.TRAIN);
        MatrixF out = layer.forward(input(6, 5, 13L));
        assertEquals(7, out.numRows());
        assertEquals(5, out.numColumns());
    }

    @Test
    void backwardDoesNotAliasTheMatrixReturnedByTheFirstBranch() {
        RetainingLayer probe = new RetainingLayer(4, 3);
        ParallelBranches layer = new ParallelBranches(
                List.of(probe),
                List.of(new Hidden(4, 4, "a1")));
        layer.setMode(NetworkMode.TRAIN);
        layer.forward(input(4, 3, 14L));
        layer.backward(input(8, 3, 15L));
        assertFalse(probe.wasMutated(), "the accumulator overwrote the matrix branch 0 returned");
    }

    @Test
    void backwardReturnsNullInInferenceMode() {
        ParallelBranches layer = new ParallelBranches(List.of(new Hidden(4, 4, "i1")));
        layer.setMode(NetworkMode.INFER);
        layer.forward(input(4, 3, 16L));
        assertNull(layer.backward(input(4, 3, 17L)));
    }

    /** A layer that keeps a reference to the gradient it returns, like Dropout does. */
    private static final class RetainingLayer extends AbstractLayer {

        private final int rows;
        private final int cols;
        private MatrixF returned;
        private MatrixF snapshot;

        RetainingLayer(int rows, int cols) {
            this.rows = rows;
            this.cols = cols;
        }

        @Override
        public MatrixF forward(MatrixF in) {
            return in;
        }

        @Override
        public MatrixF backward(MatrixF grads) {
            returned = input(rows, cols, 18L);
            snapshot = returned.copy();
            return returned;
        }

        boolean wasMutated() {
            for (int c = 0; c < cols; ++c) {
                for (int r = 0; r < rows; ++r) {
                    if (returned.getUnsafe(r, c) != snapshot.getUnsafe(r, c)) {
                        return true;
                    }
                }
            }
            return false;
        }
    }
}
