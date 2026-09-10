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

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Field;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

/**
 * Verifies a layer's {@code backward()} against central differences of its
 * {@code forward()}.
 *
 * <p>The scalar test function is {@code L(y) = sum_ij w_ij * y_ij} for a fixed
 * {@code w}, so that {@code dL/dy} is exactly the {@code w} handed to
 * {@code backward()}.
 */
final class GradientCheck {

    /** Step size for the central difference; float inputs do not tolerate less. */
    static final float H = 1e-2f;

    /** Deterministic test matrix, so a failure is reproducible rather than a flake. */
    static MatrixF input(int rows, int cols, long seed) {
        return Matrices.randomUniformF(rows, cols, -2.0f, 2.0f, seed);
    }

    /** The scalar test function, accumulated in double to keep the differences clean. */
    static double dot(MatrixF w, MatrixF y) {
        double sum = 0.0;
        for (int c = 0; c < y.numColumns(); ++c) {
            for (int r = 0; r < y.numRows(); ++r) {
                sum += (double) w.getUnsafe(r, c) * y.getUnsafe(r, c);
            }
        }
        return sum;
    }

    /**
     * Relative error with an absolute floor, so that gradients near zero do not
     * blow the ratio up.
     */
    static double relativeError(double a, double b) {
        double scale = Math.max(1e-3, Math.max(Math.abs(a), Math.abs(b)));
        return Math.abs(a - b) / scale;
    }

    /**
     * Asserts that the layer's input gradient matches central differences.
     *
     * <p>Runs with a learning rate of zero so that no parameter moves between the
     * analytic and the numerical pass.
     */
    static void assertInputGradient(Layer layer, MatrixF x, MatrixF w, double tolerance) {
        layer.setMode(NetworkMode.TRAIN);
        layer.forward(x.copy());
        MatrixF analytic = layer.backward(w.copy());

        double worst = 0.0;
        int worstRow = -1;
        int worstCol = -1;
        for (int c = 0; c < x.numColumns(); ++c) {
            for (int r = 0; r < x.numRows(); ++r) {
                double numeric = (perturbedLoss(layer, x, w, r, c, H) - perturbedLoss(layer, x, w, r, c, -H))
                        / (2.0 * H);
                double error = relativeError(numeric, analytic.getUnsafe(r, c));
                if (error > worst) {
                    worst = error;
                    worstRow = r;
                    worstCol = c;
                }
            }
        }
        double maxError = worst;
        int row = worstRow;
        int col = worstCol;
        assertTrue(maxError <= tolerance, () -> "max relative error " + maxError + " at [" + row + "," + col
                + "] exceeds tolerance " + tolerance);
    }

    /** Every layer gets a copy: several of them modify their argument in place. */
    private static double perturbedLoss(Layer layer, MatrixF x, MatrixF w, int row, int col, float delta) {
        MatrixF perturbed = x.copy();
        perturbed.setUnsafe(row, col, x.getUnsafe(row, col) + delta);
        layer.setMode(NetworkMode.TRAIN);
        return dot(w, layer.forward(perturbed));
    }

    /** An Sgd over one layer's parameters, for the tests that train a bare layer. */
    static Sgd sgdOver(Layer layer, float rate) {
        Sgd sgd = new Sgd(rate);
        for (Parameter p : layer.parameters()) {
            sgd.add(p);
        }
        return sgd;
    }

    /** Reads a private field, for the layers that cache state the contract depends on. */
    @SuppressWarnings("unchecked")
    static <T> T field(Object target, String name) throws ReflectiveOperationException {
        Field f = target.getClass().getDeclaredField(name);
        f.setAccessible(true);
        return (T) f.get(target);
    }

    /**
     * Reads a matrix-valued field, unwrapping a {@link Parameter} so that a trainable
     * parameter and a plain cached matrix read the same. The result is the live buffer,
     * which the callers that perturb a parameter depend on.
     */
    static MatrixF value(Object target, String name) throws ReflectiveOperationException {
        Object f = field(target, name);
        return f instanceof Parameter p ? p.value() : (MatrixF) f;
    }

    /** Reads the gradient buffer of a parameter field. */
    static MatrixF grad(Object target, String name) throws ReflectiveOperationException {
        return GradientCheck.<Parameter>field(target, name).grad();
    }

    private GradientCheck() {
        throw new AssertionError();
    }
}
