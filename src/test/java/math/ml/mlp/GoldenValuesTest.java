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

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

// The rest of the suite compares a run against another run of the same build, which
// catches nondeterminism but not a change in the numbers. These values were recorded
// once and must not move: they are what says an optimization was only an optimization.
//
// Deliberately free of Hidden, and therefore of sgemm: MKL varies its reduction order
// unless MKL_CBWR is set (see Audit.md section 2), which would make a golden flake.
class GoldenValuesTest {

    // JUnit compares floats by their bits, so this also pins the sign of every zero
    private static final float[] RELU_FORWARD = { 0.9278126f, 1.4883463f, 0.0f, 1.1772192f, 0.0f, 0.0f, 1.7814262f,
            0.0f, 0.100913525f, 0.24111843f, 1.725327f, 0.0f };
    private static final float[] RELU_BACKWARD = { 0.4644438f, -0.87002015f, 0.0f, -0.21795785f, 0.0f, -0.0f,
            0.19703293f, 0.0f, -0.170807f, 0.5453292f, 0.8252708f, 0.0f };

    private static final float[] GELU_FORWARD = { 0.7636923f, 1.3864276f, -0.15399379f, 1.0362697f, -0.1594734f,
            -0.1700288f, 1.7146833f, -0.11666041f, 0.05451249f, 0.1435291f, 1.6523418f, -0.16358097f };
    private static final float[] GELU_BACKWARD = { 0.49397084f, -0.9814203f, 0.09831916f, -0.24303813f, 0.032655396f,
            0.002372418f, 0.21844873f, -0.007960587f, -0.09910962f, 0.37556133f, 0.91916883f, 0.025021153f };

    private static final float[] SIGMOID_FORWARD = { 0.7166313f, 0.81582993f, 0.37805596f, 0.76444745f, 0.36739308f,
            0.31866154f, 0.8558729f, 0.20199989f, 0.525207f, 0.5599893f, 0.8488137f, 0.3572921f };
    private static final float[] SIGMOID_BACKWARD = { 0.094315015f, -0.1307218f, 0.17254515f, -0.039247133f,
            0.071236946f, -0.16133346f, 0.02430489f, 0.00998053f, -0.042593222f, 0.13436982f, 0.105906166f,
            0.070422344f };

    private static final float[] MASK_SEED_SEVEN = { 1.6666666f, 1.6666666f, 0.0f, 1.6666666f, 1.6666666f, 1.6666666f,
            1.6666666f, 0.0f, 1.6666666f, 0.0f, 1.6666666f, 1.6666666f };

    private static final float[] CHAIN_LOSSES = { 4.7095165f, 4.754839f, 5.247073f, 5.1949325f };

    @Test
    void reluIsUnchanged() {
        assertActivation(new Relu(), RELU_FORWARD, RELU_BACKWARD);
    }

    @Test
    void geluIsUnchanged() {
        assertActivation(new Gelu(), GELU_FORWARD, GELU_BACKWARD);
    }

    @Test
    void sigmoidIsUnchanged() {
        assertActivation(new Sigmoid(), SIGMOID_FORWARD, SIGMOID_BACKWARD);
    }

    @Test
    void theDropoutMaskForSeedSevenIsUnchanged() {
        Dropout dropout = new Dropout(0.4f, 7L);
        dropout.setMode(NetworkMode.TRAIN);
        MatrixF ones = Matrices.createF(4, 3);
        Arrays.fill(ones.getArrayUnsafe(), 1.0f);
        assertExactly(MASK_SEED_SEVEN, dropout.forward(ones));
    }

    @Test
    void theGemmFreeChainIsUnchanged() {
        List<Float> losses = new ArrayList<>();
        SigmoidBCELoss loss = new SigmoidBCELoss();
        loss.registerLossCallback(l -> losses.add(Matrices.colsAverage(l).toScalar()));

        Net net = new Net();
        net.add(new Dropout(0.3f, 31L));
        net.add(new Relu());
        net.add(new Dropout(0.25f, 32L));
        net.add(new Gelu());
        net.add(loss);

        MatrixF expected = Matrices.randomUniformF(6, 5, 0.0f, 1.0f, 33L);
        for (int batch = 0; batch < CHAIN_LOSSES.length; ++batch) {
            net.train(Matrices.randomUniformF(6, 5, -2.0f, 2.0f, 40L + batch), expected, 0.01f);
        }

        assertEquals(CHAIN_LOSSES.length, losses.size());
        for (int i = 0; i < CHAIN_LOSSES.length; ++i) {
            assertEquals(CHAIN_LOSSES[i], losses.get(i).floatValue(), "loss of batch " + i);
        }
    }

    private static void assertActivation(Activation activation, float[] forward, float[] backward) {
        MatrixF x = Matrices.randomUniformF(4, 3, -2.0f, 2.0f, 21L);
        MatrixF g = Matrices.randomUniformF(4, 3, -1.0f, 1.0f, 22L);
        activation.setMode(NetworkMode.TRAIN);
        assertExactly(forward, activation.forward(x.copy()));
        assertExactly(backward, activation.backward(g.copy(), 0.0f));
    }

    private static void assertExactly(float[] expected, MatrixF actual) {
        float[] a = actual.getArrayUnsafe();
        assertEquals(expected.length, a.length);
        for (int i = 0; i < expected.length; ++i) {
            assertEquals(expected[i], a[i], "element " + i);
        }
    }

    private static final class Net extends AbstractNetwork {
    }
}
