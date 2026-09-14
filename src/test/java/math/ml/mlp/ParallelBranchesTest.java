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
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
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
                List.of(new Hidden(6, 4, "p1", 103L), new Sigmoid()),
                List.of(new Hidden(6, 4, "p2", 104L)));
        assertInputGradient(layer, input(6, 5, 11L), input(8, 5, 12L), 3e-2);
    }

    @Test
    void backwardMatchesNumericalInputGradientWithoutAnActivation() {
        // branch 0 above contains a layer that may mutate, neither branch here does,
        // so the two exercise the copied and the uncopied path through the same code
        ParallelBranches layer = new ParallelBranches(
                List.of(new Hidden(6, 4, "p3", 105L)),
                List.of(new Hidden(6, 4, "p4", 106L)));
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
    void anEmptyBranchListIsRejected() {
        assertThrows(IllegalArgumentException.class, ParallelBranches::new);
    }

    @Test
    void backwardReturnsNullInInferenceMode() {
        ParallelBranches layer = new ParallelBranches(List.of(new Hidden(4, 4, "i1")));
        layer.setMode(NetworkMode.INFER);
        layer.forward(input(4, 3, 16L));
        assertNull(layer.backward(input(4, 3, 17L)));
    }

    @Test
    void theGradientSurvivesALaterForward() {
        // the branch heights add up to the input height on purpose, so the stacked output and
        // the summed gradient have the same shape: one buffer for both would go unnoticed at
        // any other pair of shapes, because a differing shape reallocates
        ParallelBranches layer = new ParallelBranches(
                List.of(new Hidden(8, 4, "g1", 107L)),
                List.of(new Hidden(8, 4, "g2", 108L)));
        layer.setMode(NetworkMode.TRAIN);
        layer.forward(input(8, 5, 41L));
        MatrixF gradients = layer.backward(input(8, 5, 42L));
        float[] taken = gradients.copy().getArrayUnsafe();
        for (int i = 0; i < 20; ++i) {
            layer.forward(input(8, 5, 43L + i));
        }
        assertArrayEquals(taken, gradients.getArrayUnsafe());
    }

    @Test
    void aChangedShapeReallocatesTheReusedBuffers() {
        ParallelBranches layer = new ParallelBranches(
                List.of(new Hidden(6, 4, "r1", 109L)),
                List.of(new Hidden(6, 3, "r2", 110L)));
        ParallelBranches fresh = new ParallelBranches(
                List.of(new Hidden(6, 4, "r1", 109L)),
                List.of(new Hidden(6, 3, "r2", 110L)));
        layer.setMode(NetworkMode.TRAIN);
        fresh.setMode(NetworkMode.TRAIN);

        layer.forward(input(6, 5, 51L));
        layer.backward(input(7, 5, 52L));

        MatrixF wide = layer.forward(input(6, 9, 53L));
        assertEquals(7, wide.numRows());
        assertEquals(9, wide.numColumns());
        assertArrayEquals(fresh.forward(input(6, 9, 53L)).getArrayUnsafe(), wide.getArrayUnsafe(), 1e-5f);

        MatrixF wideGradients = layer.backward(input(7, 9, 54L));
        assertEquals(6, wideGradients.numRows());
        assertEquals(9, wideGradients.numColumns());
        assertArrayEquals(fresh.backward(input(7, 9, 54L)).getArrayUnsafe(), wideGradients.getArrayUnsafe(), 1e-5f);

        MatrixF narrow = layer.forward(input(6, 5, 55L));
        assertEquals(5, narrow.numColumns());
        assertArrayEquals(fresh.forward(input(6, 5, 55L)).getArrayUnsafe(), narrow.getArrayUnsafe(), 1e-5f);
    }

    @Test
    void theSummedGradientIsTheSumOfBothSlicesAndNotOneOfThem() {
        ParallelBranches layer = new ParallelBranches(List.of(new PassThrough()), List.of(new PassThrough()));
        layer.setMode(NetworkMode.TRAIN);
        layer.forward(input(4, 3, 61L));
        MatrixF grads = input(8, 3, 62L);
        MatrixF summed = layer.backward(grads);
        for (int c = 0; c < 3; ++c) {
            for (int r = 0; r < 4; ++r) {
                assertEquals(grads.getUnsafe(r, c) + grads.getUnsafe(r + 4, c), summed.getUnsafe(r, c), 1e-5f);
            }
        }
    }

    @Test
    void oneBranchCannotOverwriteTheInputAnotherBranchWasGiven() {
        InPlaceProbe probe = new InPlaceProbe();
        ParallelBranches layer = new ParallelBranches(
                List.of(probe),
                List.of(new Hidden(4, 4, "c1", 111L)));
        layer.setMode(NetworkMode.TRAIN);
        layer.forward(input(4, 3, 63L));
        assertTrue(probe.stillHoldsWhatItWrote(), "a later branch refilled the matrix branch 0 was given");
    }

    /** Returns what it is handed, so the summed gradient is the two halves and nothing else. */
    private static final class PassThrough extends AbstractLayer {

        @Override
        public MatrixF forward(MatrixF in) {
            return in;
        }

        @Override
        public MatrixF backward(MatrixF grads) {
            return grads;
        }
    }

    /** Writes into its input and keeps the reference, which a shared buffer shows up in. */
    private static final class InPlaceProbe extends AbstractLayer {

        private MatrixF seen;
        private MatrixF snapshot;

        @Override
        public boolean mutatesInput() {
            return true;
        }

        @Override
        public MatrixF forward(MatrixF in) {
            seen = in.scaleInplace(2.0f);
            snapshot = seen.copy();
            return seen;
        }

        @Override
        public MatrixF backward(MatrixF grads) {
            return grads;
        }

        boolean stillHoldsWhatItWrote() {
            for (int c = 0; c < seen.numColumns(); ++c) {
                for (int r = 0; r < seen.numRows(); ++r) {
                    if (seen.getUnsafe(r, c) != snapshot.getUnsafe(r, c)) {
                        return false;
                    }
                }
            }
            return true;
        }
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
