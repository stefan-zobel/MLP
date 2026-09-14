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
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Field;

import org.junit.jupiter.api.Test;

import math.dl.GELU;
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

    // Above 2^18 elements the forward and backward loops are split across cores, and the
    // fixtures elsewhere are all far too small to reach that path. A split that is off by
    // one element, or that leaves a gap between two chunks, shows up here and nowhere else.
    @Test
    void theThreadedActivationPathIsExactlyTheSequentialOne() throws ReflectiveOperationException {
        MatrixF x = Matrices.randomUniformF(512, 1024, -4.0f, 4.0f, 91L);
        MatrixF g = Matrices.randomUniformF(512, 1024, -0.5f, 0.5f, 93L);
        // without this the fixture would go on passing after someone raised the threshold
        // past it, comparing the sequential path against itself and proving nothing
        Field threshold = Activation.class.getDeclaredField("PARALLEL_THRESHOLD");
        threshold.setAccessible(true);
        assertTrue(x.getArrayUnsafe().length >= threshold.getInt(null), "the fixture no longer reaches the split");
        Gelu gelu = new Gelu();
        gelu.setMode(NetworkMode.TRAIN);

        MatrixF forward = gelu.forward(x);
        MatrixF backward = gelu.backward(g);

        float[] in = x.getArrayUnsafe();
        float[] grads = g.getArrayUnsafe();
        float[] f = forward.getArrayUnsafe();
        float[] b = backward.getArrayUnsafe();
        for (int i = 0; i < in.length; ++i) {
            assertEquals(Float.floatToRawIntBits(GELU.geluF(in[i])), Float.floatToRawIntBits(f[i]), "forward at " + i);
            assertEquals(Float.floatToRawIntBits(grads[i] * GELU.dgeluF_dx(in[i])), Float.floatToRawIntBits(b[i]),
                    "backward at " + i);
        }
    }

    // The output buffer is reused across steps, so a changing batch size has to reallocate it.
    // A validation pass at a different batch, or a final short batch, would otherwise read a
    // stale buffer of the wrong length.
    @Test
    void aChangedShapeReallocatesTheReusedBuffer() {
        Relu relu = new Relu();
        relu.setMode(NetworkMode.TRAIN);

        MatrixF wide = relu.forward(Matrices.randomUniformF(4, 6, 1.0f, 2.0f, 95L));
        assertEquals(4, wide.numRows());
        assertEquals(6, wide.numColumns());

        MatrixF narrow = relu.forward(Matrices.randomUniformF(4, 2, 1.0f, 2.0f, 96L));
        assertEquals(4, narrow.numRows());
        assertEquals(2, narrow.numColumns());

        MatrixF x = Matrices.randomUniformF(4, 2, 1.0f, 2.0f, 97L);
        MatrixF again = relu.forward(x);
        assertSame(narrow, again, "the buffer is reused while the shape holds");
        for (int i = 0; i < x.getArrayUnsafe().length; ++i) {
            assertEquals(x.getArrayUnsafe()[i], again.getArrayUnsafe()[i], "element " + i);
        }
    }
}
