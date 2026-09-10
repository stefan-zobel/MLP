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
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import net.jamu.matrix.MatrixF;

class SoftmaxTest {

    @Test
    void backwardMatchesNumericalInputGradient() {
        assertInputGradient(new Softmax(), input(6, 4, 51L), input(6, 4, 52L), 2e-2);
    }

    @Test
    void everyColumnIsAProbabilityDistribution() {
        Softmax layer = new Softmax();
        layer.setMode(NetworkMode.TRAIN);
        MatrixF out = layer.forward(input(7, 5, 53L));
        for (int c = 0; c < out.numColumns(); ++c) {
            double sum = 0.0;
            for (int r = 0; r < out.numRows(); ++r) {
                float p = out.getUnsafe(r, c);
                assertTrue(p > 0.0f && p < 1.0f, "probability out of range: " + p);
                sum += p;
            }
            assertEquals(1.0, sum, 1e-5, "column " + c + " must sum to 1");
        }
    }

    @Test
    void largeLogitsDoNotOverflow() {
        Softmax layer = new Softmax();
        layer.setMode(NetworkMode.TRAIN);
        MatrixF logits = net.jamu.matrix.Matrices.createF(3, 1);
        logits.setUnsafe(0, 0, 1000.0f);
        logits.setUnsafe(1, 0, 999.0f);
        logits.setUnsafe(2, 0, -1000.0f);

        MatrixF out = layer.forward(logits);
        double sum = 0.0;
        for (int r = 0; r < 3; ++r) {
            assertTrue(Float.isFinite(out.getUnsafe(r, 0)), "row " + r + " is not finite");
            sum += out.getUnsafe(r, 0);
        }
        assertEquals(1.0, sum, 1e-5);
    }

    @Test
    void backwardReturnsNullInInferenceMode() {
        Softmax layer = new Softmax();
        layer.setMode(NetworkMode.INFER);
        layer.forward(input(4, 3, 54L));
        assertNull(layer.backward(input(4, 3, 55L)));
    }
}
