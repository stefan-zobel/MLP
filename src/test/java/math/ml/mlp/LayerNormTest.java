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
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;

import org.junit.jupiter.api.Test;

import net.jamu.matrix.MatrixF;

class LayerNormTest {

    private static final int FEATURES = 6;
    private static final int BATCH = 5;

    @Test
    void backwardMatchesNumericalInputGradient() {
        assertInputGradient(new LayerNorm(FEATURES), input(FEATURES, BATCH, 701L), input(FEATURES, BATCH, 702L), 3e-2);
    }

    @Test
    void everySampleIsNormalizedOverItsFeatures() {
        // the property a gradient check cannot catch: if the axes were swapped the
        // gradients would still be right, but the rows would be normalized instead
        LayerNorm layer = new LayerNorm(FEATURES);
        layer.setMode(NetworkMode.TRAIN);
        MatrixF out = layer.forward(input(FEATURES, BATCH, 703L));

        for (int c = 0; c < BATCH; ++c) {
            double mean = 0.0;
            for (int r = 0; r < FEATURES; ++r) {
                mean += out.getUnsafe(r, c);
            }
            mean /= FEATURES;
            double var = 0.0;
            for (int r = 0; r < FEATURES; ++r) {
                double d = out.getUnsafe(r, c) - mean;
                var += d * d;
            }
            var /= FEATURES;
            assertEquals(0.0, mean, 1e-4, "column " + c + " must have zero mean");
            assertEquals(1.0, var, 1e-3, "column " + c + " must have unit variance");
        }
    }

    @Test
    void trainingAndInferenceAgree() {
        // BatchNorm deliberately differs between the modes; LayerNorm must not
        MatrixF x = input(FEATURES, BATCH, 704L);
        LayerNorm layer = new LayerNorm(FEATURES);

        layer.setMode(NetworkMode.TRAIN);
        MatrixF trained = layer.forward(x.copy());
        layer.setMode(NetworkMode.INFER);
        MatrixF inferred = layer.forward(x.copy());

        for (int c = 0; c < BATCH; ++c) {
            for (int r = 0; r < FEATURES; ++r) {
                assertEquals(trained.getUnsafe(r, c), inferred.getUnsafe(r, c), 1e-6f);
            }
        }
    }

    @Test
    void oneSampleDoesNotInfluenceAnother() {
        // the actual difference to BatchNorm: no cross-sample coupling
        LayerNorm layer = new LayerNorm(FEATURES);
        layer.setMode(NetworkMode.TRAIN);
        MatrixF x = input(FEATURES, BATCH, 705L);
        MatrixF first = layer.forward(x.copy());

        MatrixF changed = x.copy();
        for (int r = 0; r < FEATURES; ++r) {
            changed.setUnsafe(r, BATCH - 1, changed.getUnsafe(r, BATCH - 1) + 10.0f);
        }
        MatrixF second = layer.forward(changed);

        for (int c = 0; c < BATCH - 1; ++c) {
            for (int r = 0; r < FEATURES; ++r) {
                assertEquals(first.getUnsafe(r, c), second.getUnsafe(r, c), 1e-5f,
                        "column " + c + " changed although only the last sample did");
            }
        }
    }

    @Test
    void gammaAndBetaUpdatesMatchNumericalGradients() throws Exception {
        MatrixF x = input(FEATURES, BATCH, 706L);
        MatrixF w = input(FEATURES, BATCH, 707L);

        LayerNorm layer = new LayerNorm(FEATURES);
        layer.setMode(NetworkMode.TRAIN);
        layer.forward(x.copy());
        layer.backward(w.copy());

        // the gradient the optimizer will consume, read directly rather than recovered
        // from how far an SGD step moved the parameter
        MatrixF dGamma = grad(layer, "gamma");
        MatrixF dBeta = grad(layer, "beta");
        for (int r = 0; r < FEATURES; ++r) {
            // the buffers hold the mean over the batch; the numerical check sums
            double analyticGamma = dGamma.getUnsafe(r, 0) * BATCH;
            double analyticBeta = dBeta.getUnsafe(r, 0) * BATCH;
            assertTrue(relativeError(numericalParameterGradient(x, w, "gamma", r), analyticGamma) <= 2e-2,
                    "gamma gradient mismatch in row " + r);
            assertTrue(relativeError(numericalParameterGradient(x, w, "beta", r), analyticBeta) <= 2e-2,
                    "beta gradient mismatch in row " + r);
        }
    }

    @Test
    void parametersSurviveAWriteReadRoundTrip() throws Exception {
        LayerNorm original = new LayerNorm(FEATURES);
        // the optimizer step is what moves gamma and beta at all; without it both
        // layers would still hold gamma 1 and beta 0 and the test would pass vacuously
        Sgd sgd = GradientCheck.sgdOver(original, 0.05f);
        for (int i = 0; i < 20; ++i) {
            original.setMode(NetworkMode.TRAIN);
            original.forward(input(FEATURES, BATCH, 800L + i));
            original.backward(input(FEATURES, BATCH, 900L + i));
            sgd.step();
        }

        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        original.writeParameters(buffer);
        LayerNorm restored = new LayerNorm(FEATURES);
        restored.readParameters(new ByteArrayInputStream(buffer.toByteArray()));

        for (String name : new String[] { "gamma", "beta" }) {
            MatrixF a = value(original, name);
            MatrixF b = value(restored, name);
            for (int r = 0; r < FEATURES; ++r) {
                assertEquals(a.getUnsafe(r, 0), b.getUnsafe(r, 0), 0.0f, name + " differs in row " + r);
            }
        }

        MatrixF x = input(FEATURES, BATCH, 708L);
        original.setMode(NetworkMode.INFER);
        restored.setMode(NetworkMode.INFER);
        MatrixF expected = original.forward(x.copy());
        MatrixF actual = restored.forward(x.copy());
        for (int c = 0; c < BATCH; ++c) {
            for (int r = 0; r < FEATURES; ++r) {
                assertEquals(expected.getUnsafe(r, c), actual.getUnsafe(r, c), 0.0f);
            }
        }
    }

    @Test
    void aMisSizedInputIsRejected() {
        LayerNorm layer = new LayerNorm(FEATURES);
        layer.setMode(NetworkMode.TRAIN);
        assertThrows(IllegalArgumentException.class, () -> layer.forward(input(FEATURES + 1, BATCH, 709L)));
    }

    @Test
    void backwardReturnsNullInInferenceMode() {
        LayerNorm layer = new LayerNorm(FEATURES);
        layer.setMode(NetworkMode.INFER);
        layer.forward(input(FEATURES, BATCH, 710L));
        assertNull(layer.backward(input(FEATURES, BATCH, 711L)));
    }

    @Test
    void storeParametersIsANoOpWithoutStoringEnabled() {
        new LayerNorm(FEATURES).storeParameters();
        new LayerNorm(FEATURES, 1e-5f).storeParameters();
    }

    private static double numericalParameterGradient(MatrixF x, MatrixF w, String name, int row) throws Exception {
        return (lossWithParameter(x, w, name, row, H) - lossWithParameter(x, w, name, row, -H)) / (2.0 * H);
    }

    private static double lossWithParameter(MatrixF x, MatrixF w, String name, int row, float delta) throws Exception {
        LayerNorm probe = new LayerNorm(FEATURES);
        probe.setMode(NetworkMode.TRAIN);
        MatrixF parameter = value(probe, name);
        parameter.setUnsafe(row, 0, parameter.getUnsafe(row, 0) + delta);
        return dot(w, probe.forward(x.copy()));
    }
}
