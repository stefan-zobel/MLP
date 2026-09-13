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

import org.junit.jupiter.api.Test;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

/**
 * Central differences are only meaningful on a smooth function, so Relu is
 * checked away from its kink at zero rather than across it.
 */
class ActivationTest {

    @Test
    void sigmoidBackwardMatchesNumericalGradient() {
        assertInputGradient(new Sigmoid(), input(6, 4, 71L), input(6, 4, 72L), 2e-2);
    }

    @Test
    void geluBackwardMatchesNumericalGradient() {
        assertInputGradient(new Gelu(), input(6, 4, 73L), input(6, 4, 74L), 2e-2);
    }

    @Test
    void reluBackwardMatchesNumericalGradientOnThePositiveBranch() {
        MatrixF positive = Matrices.randomUniformF(6, 4, 1.0f, 3.0f, 75L);
        assertInputGradient(new Relu(), positive, input(6, 4, 76L), 2e-2);
    }

    @Test
    void reluBackwardMatchesNumericalGradientOnTheNegativeBranch() {
        MatrixF negative = Matrices.randomUniformF(6, 4, -3.0f, -1.0f, 77L);
        assertInputGradient(new Relu(), negative, input(6, 4, 78L), 2e-2);
    }

    @Test
    void reluPassesPositivesAndBlocksNegatives() {
        Relu relu = new Relu();
        relu.setMode(NetworkMode.TRAIN);
        MatrixF x = Matrices.createF(2, 1);
        x.setUnsafe(0, 0, 2.5f);
        x.setUnsafe(1, 0, -2.5f);

        MatrixF y = relu.forward(x.copy());
        assertEquals(2.5f, y.getUnsafe(0, 0), 1e-6f);
        assertEquals(0.0f, y.getUnsafe(1, 0), 1e-6f);

        MatrixF ones = Matrices.createF(2, 1);
        ones.setUnsafe(0, 0, 1.0f);
        ones.setUnsafe(1, 0, 1.0f);
        MatrixF g = relu.backward(ones);
        assertEquals(1.0f, g.getUnsafe(0, 0), 1e-6f, "gradient passes where x > 0");
        assertEquals(0.0f, g.getUnsafe(1, 0), 1e-6f, "gradient is blocked where x < 0");
    }

    @Test
    void sigmoidMapsIntoTheOpenUnitInterval() {
        Sigmoid sigmoid = new Sigmoid();
        sigmoid.setMode(NetworkMode.INFER);
        MatrixF y = sigmoid.forward(Matrices.randomUniformF(8, 4, -20.0f, 20.0f, 79L));
        for (int c = 0; c < y.numColumns(); ++c) {
            for (int r = 0; r < y.numRows(); ++r) {
                float p = y.getUnsafe(r, c);
                assertEquals(true, p >= 0.0f && p <= 1.0f, "sigmoid out of range: " + p);
            }
        }
    }

    @Test
    void backwardReturnsNullInInferenceMode() {
        Relu relu = new Relu();
        relu.setMode(NetworkMode.INFER);
        relu.forward(input(4, 3, 80L));
        assertNull(relu.backward(input(4, 3, 81L)));
    }
}
