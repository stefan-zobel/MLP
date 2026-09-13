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
import static math.ml.mlp.GradientCheck.grad;
import static math.ml.mlp.GradientCheck.input;
import static math.ml.mlp.GradientCheck.relativeError;
import static math.ml.mlp.GradientCheck.value;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;

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
        MatrixF x = input(features, batch, 3L);
        MatrixF w = input(features, batch, 4L);

        BatchNorm layer = new BatchNorm(features);
        layer.setMode(NetworkMode.TRAIN);
        layer.forward(x.copy());
        layer.backward(w.copy());

        // the gradient the optimizer will consume, read directly rather than recovered
        // from how far an SGD step moved the parameter
        MatrixF dGamma = grad(layer, "gamma");
        MatrixF dBeta = grad(layer, "beta");
        for (int r = 0; r < features; ++r) {
            // the buffers hold the mean over the batch; the numerical check sums
            double analyticGamma = dGamma.getUnsafe(r, 0) * batch;
            double analyticBeta = dBeta.getUnsafe(r, 0) * batch;

            assertTrue(relativeError(numericalParameterGradient(features, x, w, "gamma", r), analyticGamma) <= 2e-2,
                    "gamma gradient mismatch in row " + r);
            assertTrue(relativeError(numericalParameterGradient(features, x, w, "beta", r), analyticBeta) <= 2e-2,
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
            layer.backward(Matrices.createF(features, batch));
        }
        MatrixF runningMean = value(layer, "runningMean");
        MatrixF runningVar = value(layer, "runningVar");
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
        assertTrue(layer.backward(input(3, 4, 8L)) == null);
    }

    @Test
    void aMisSizedInputIsRejectedInTrainingMode() {
        BatchNorm layer = new BatchNorm(6);
        layer.setMode(NetworkMode.TRAIN);
        assertThrows(IllegalArgumentException.class, () -> layer.forward(input(5, 4, 500L)));
    }

    @Test
    void aMisSizedInputIsRejectedInInferenceMode() {
        BatchNorm layer = new BatchNorm(6);
        layer.setMode(NetworkMode.INFER);
        assertThrows(IllegalArgumentException.class, () -> layer.forward(input(7, 4, 501L)));
    }
    @Test
    void parametersSurviveAWriteReadRoundTrip() throws Exception {
        int features = 6;
        int batch = 32;
        BatchNorm original = new BatchNorm(features);
        // train a little so gamma, beta and the running statistics all move away from
        // their initial values. Without the optimizer step this test would pass
        // vacuously, both layers still holding gamma 1 and beta 0.
        Sgd sgd = GradientCheck.sgdOver(original, 0.05f);
        for (int i = 0; i < 20; ++i) {
            original.setMode(NetworkMode.TRAIN);
            original.forward(Matrices.randomUniformF(features, batch, 2.0f, 8.0f, i));
            original.backward(input(features, batch, 200L + i));
            sgd.step();
        }

        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        original.writeParameters(buffer);

        BatchNorm restored = new BatchNorm(features);
        restored.readParameters(new ByteArrayInputStream(buffer.toByteArray()));

        for (String field : new String[] { "gamma", "beta", "runningMean", "runningVar" }) {
            MatrixF a = value(original, field);
            MatrixF b = value(restored, field);
            for (int r = 0; r < features; ++r) {
                assertEquals(a.getUnsafe(r, 0), b.getUnsafe(r, 0), 0.0f, field + " differs in row " + r);
            }
        }
    }

    @Test
    void aRestoredLayerInfersIdentically() throws Exception {
        int features = 5;
        int batch = 24;
        BatchNorm original = new BatchNorm(features);
        Sgd sgd = GradientCheck.sgdOver(original, 0.05f);
        for (int i = 0; i < 20; ++i) {
            original.setMode(NetworkMode.TRAIN);
            original.forward(Matrices.randomUniformF(features, batch, -1.0f, 4.0f, i));
            original.backward(input(features, batch, 300L + i));
            sgd.step();
        }

        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        original.writeParameters(buffer);
        BatchNorm restored = new BatchNorm(features);
        restored.readParameters(new ByteArrayInputStream(buffer.toByteArray()));

        MatrixF x = input(features, 7, 301L);
        original.setMode(NetworkMode.INFER);
        restored.setMode(NetworkMode.INFER);
        MatrixF expected = original.forward(x.copy());
        MatrixF actual = restored.forward(x.copy());

        for (int c = 0; c < x.numColumns(); ++c) {
            for (int r = 0; r < features; ++r) {
                assertEquals(expected.getUnsafe(r, c), actual.getUnsafe(r, c), 0.0f);
            }
        }
    }

    @Test
    void storeParametersIsANoOpWithoutStoringEnabled() {
        // the plain constructors must not touch the filesystem
        new BatchNorm(4).storeParameters();
        new BatchNorm(4, 1e-5f, 0.1f).storeParameters();
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
        MatrixF parameter = value(probe, name);
        parameter.setUnsafe(row, 0, parameter.getUnsafe(row, 0) + delta);
        return dot(w, probe.forward(x.copy()));
    }
}
