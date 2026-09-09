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

import static math.ml.mlp.GradientCheck.H;
import static math.ml.mlp.GradientCheck.assertInputGradient;
import static math.ml.mlp.GradientCheck.dot;
import static math.ml.mlp.GradientCheck.field;
import static math.ml.mlp.GradientCheck.input;
import static math.ml.mlp.GradientCheck.relativeError;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

class BatchNormTest {

    @Test
    void backwardMatchesNumericalInputGradient() {
        assertInputGradient(new BatchNorm(6), input(6, 8, 1L), input(6, 8, 2L), 2e-2);
    }

    @Test
    void gammaAndBetaUpdatesMatchNumericalGradients() throws Exception {
        int features = 5;
        int batch = 7;
        float learningRate = 0.1f;
        MatrixF x = input(features, batch, 3L);
        MatrixF w = input(features, batch, 4L);

        BatchNorm layer = new BatchNorm(features);
        layer.setMode(NetworkMode.TRAIN);
        MatrixF gammaBefore = GradientCheck.<MatrixF>field(layer, "gamma").copy();
        MatrixF betaBefore = GradientCheck.<MatrixF>field(layer, "beta").copy();

        layer.forward(x.copy());
        layer.backward(w.copy(), learningRate);

        MatrixF gamma = field(layer, "gamma");
        MatrixF beta = field(layer, "beta");
        for (int r = 0; r < features; ++r) {
            // the update is gamma_r -= lr * dGamma_r / m, so invert it to recover dGamma_r
            double appliedGamma = (gammaBefore.getUnsafe(r, 0) - gamma.getUnsafe(r, 0)) / learningRate * batch;
            double appliedBeta = (betaBefore.getUnsafe(r, 0) - beta.getUnsafe(r, 0)) / learningRate * batch;

            assertTrue(relativeError(numericalParameterGradient(features, x, w, "gamma", r), appliedGamma) <= 2e-2,
                    "gamma gradient mismatch in row " + r);
            assertTrue(relativeError(numericalParameterGradient(features, x, w, "beta", r), appliedBeta) <= 2e-2,
                    "beta gradient mismatch in row " + r);
        }
    }

    @Test
    void runningStatisticsConvergeToTheBatchDistribution() throws Exception {
        int features = 4;
        int batch = 64;
        BatchNorm layer = new BatchNorm(features);
        for (int i = 0; i < 500; ++i) {
            // uniform(-3, 3) shifted by 5: mean 5, variance 3
            MatrixF x = Matrices.randomUniformF(features, batch, 2.0f, 8.0f, i);
            layer.setMode(NetworkMode.TRAIN);
            layer.forward(x);
            layer.backward(Matrices.createF(features, batch), 0.0f);
        }
        MatrixF runningMean = field(layer, "runningMean");
        MatrixF runningVar = field(layer, "runningVar");
        for (int r = 0; r < features; ++r) {
            assertEquals(5.0, runningMean.getUnsafe(r, 0), 0.3, "runningMean in row " + r);
            assertEquals(3.0, runningVar.getUnsafe(r, 0), 0.5, "runningVar in row " + r);
        }
    }

    @Test
    void inferenceUsesRunningStatisticsInsteadOfTheBatch() {
        int features = 3;
        BatchNorm layer = new BatchNorm(features);
        MatrixF x = input(features, 5, 5L);

        layer.setMode(NetworkMode.INFER);
        MatrixF first = layer.forward(x.copy());
        // a completely different second batch must not change the first result
        layer.forward(input(features, 5, 6L));
        MatrixF again = layer.forward(x.copy());

        for (int c = 0; c < first.numColumns(); ++c) {
            for (int r = 0; r < features; ++r) {
                assertEquals(first.getUnsafe(r, c), again.getUnsafe(r, c), 1e-6f);
            }
        }
    }

    @Test
    void backwardReturnsNullInInferenceMode() {
        BatchNorm layer = new BatchNorm(3);
        layer.setMode(NetworkMode.INFER);
        layer.forward(input(3, 4, 7L));
        assertTrue(layer.backward(input(3, 4, 8L), 0.1f) == null);
    }

    private static double numericalParameterGradient(int features, MatrixF x, MatrixF w, String name, int row)
            throws Exception {
        double plus = lossWithParameter(features, x, w, name, row, H);
        double minus = lossWithParameter(features, x, w, name, row, -H);
        return (plus - minus) / (2.0 * H);
    }

    private static double lossWithParameter(int features, MatrixF x, MatrixF w, String name, int row, float delta)
            throws Exception {
        BatchNorm probe = new BatchNorm(features);
        probe.setMode(NetworkMode.TRAIN);
        MatrixF parameter = field(probe, name);
        parameter.setUnsafe(row, 0, parameter.getUnsafe(row, 0) + delta);
        return dot(w, probe.forward(x.copy()));
    }
}
