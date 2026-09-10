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

import static math.ml.mlp.GradientCheck.input;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

/** The mask is drawn from an unseeded generator, so these are statistical bounds. */
class DropoutTest {

    @Test
    void invertedDropoutPreservesTheExpectedActivation() {
        float rate = 0.3f;
        int rows = 40;
        int cols = 40;
        int trials = 200;
        Dropout layer = new Dropout(rate);
        layer.setMode(NetworkMode.TRAIN);

        double sum = 0.0;
        for (int i = 0; i < trials; ++i) {
            MatrixF y = layer.forward(ones(rows, cols));
            for (int c = 0; c < cols; ++c) {
                for (int r = 0; r < rows; ++r) {
                    sum += y.getUnsafe(r, c);
                }
            }
        }
        assertEquals(1.0, sum / (trials * (double) rows * cols), 0.02);
    }

    @Test
    void backwardZeroesExactlyThePositionsForwardDropped() {
        int rows = 20;
        int cols = 8;
        Dropout layer = new Dropout(0.4f);
        layer.setMode(NetworkMode.TRAIN);

        MatrixF activations = layer.forward(ones(rows, cols));
        MatrixF gradients = layer.backward(ones(rows, cols), 0.0f);

        float scale = 1.0f / (1.0f - 0.4f);
        int dropped = 0;
        for (int c = 0; c < cols; ++c) {
            for (int r = 0; r < rows; ++r) {
                boolean wasDropped = activations.getUnsafe(r, c) == 0.0f;
                if (wasDropped) {
                    ++dropped;
                    assertEquals(0.0f, gradients.getUnsafe(r, c), 0.0f, "gradient must be masked at [" + r + "," + c + "]");
                } else {
                    assertEquals(scale, gradients.getUnsafe(r, c), 1e-5f, "gradient must be scaled at [" + r + "," + c + "]");
                }
            }
        }
        assertTrue(dropped > 0, "the test is meaningless if nothing was dropped");
    }

    @Test
    void inferenceLeavesTheInputUntouched() {
        Dropout layer = new Dropout(0.5f);
        layer.setMode(NetworkMode.INFER);
        MatrixF x = input(6, 4, 61L);
        MatrixF original = x.copy();

        MatrixF out = layer.forward(x);
        assertSame(x, out);
        for (int c = 0; c < x.numColumns(); ++c) {
            for (int r = 0; r < x.numRows(); ++r) {
                assertEquals(original.getUnsafe(r, c), out.getUnsafe(r, c), 0.0f);
            }
        }
    }

    @Test
    void aZeroRateIsAPassThrough() {
        Dropout layer = new Dropout(0.0f);
        layer.setMode(NetworkMode.TRAIN);
        MatrixF x = input(6, 4, 62L);
        MatrixF original = x.copy();

        MatrixF out = layer.forward(x);
        for (int c = 0; c < x.numColumns(); ++c) {
            for (int r = 0; r < x.numRows(); ++r) {
                assertEquals(original.getUnsafe(r, c), out.getUnsafe(r, c), 0.0f);
            }
        }
    }

    @Test
    void theMaskIsRebuiltWhenTheBatchSizeChanges() {
        Dropout layer = new Dropout(0.4f);
        layer.setMode(NetworkMode.TRAIN);
        layer.forward(ones(10, 8));
        // a smaller batch must not read stale mask bits
        MatrixF activations = layer.forward(ones(10, 3));
        MatrixF gradients = layer.backward(ones(10, 3), 0.0f);
        for (int c = 0; c < 3; ++c) {
            for (int r = 0; r < 10; ++r) {
                boolean wasDropped = activations.getUnsafe(r, c) == 0.0f;
                assertEquals(wasDropped, gradients.getUnsafe(r, c) == 0.0f, "mask disagrees at [" + r + "," + c + "]");
            }
        }
    }

    @Test
    void backwardReturnsNullInInferenceMode() {
        Dropout layer = new Dropout(0.3f);
        layer.setMode(NetworkMode.INFER);
        layer.forward(input(4, 3, 63L));
        assertNull(layer.backward(input(4, 3, 64L), 0.1f));
    }

    private static MatrixF ones(int rows, int cols) {
        MatrixF m = Matrices.createF(rows, cols);
        for (int c = 0; c < cols; ++c) {
            for (int r = 0; r < rows; ++r) {
                m.setUnsafe(r, c, 1.0f);
            }
        }
        return m;
    }
}
