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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

class BinaryCrossEntropyLossTest {

    private static final int ROWS = 6;
    private static final int COLS = 3;

    @Test
    void forwardReturnsTheUnfusedGradient() {
        MatrixF pred = probabilities();
        MatrixF targets = alternatingTargets();

        MatrixF gradients = forward(pred, targets);
        for (int c = 0; c < COLS; ++c) {
            for (int r = 0; r < ROWS; ++r) {
                float p = pred.getUnsafe(r, c);
                float t = targets.getUnsafe(r, c);
                assertEquals((p - t) / (p * (1.0f - p)), gradients.getUnsafe(r, c), 1e-3f);
            }
        }
    }

    @Test
    void reportedLossIsNormalisedByTheOutputDimension() {
        MatrixF pred = probabilities();
        MatrixF targets = alternatingTargets();

        MatrixF[] captured = new MatrixF[1];
        BinaryCrossEntropyLoss loss = new BinaryCrossEntropyLoss();
        loss.setMode(NetworkMode.TRAIN);
        loss.registerBatchExpectedValuesProvider(n -> targets);
        loss.registerLossCallback(l -> captured[0] = l);
        loss.forward(pred);

        assertNotNull(captured[0]);
        assertEquals(1, captured[0].numRows());
        assertEquals(COLS, captured[0].numColumns());
        for (int c = 0; c < COLS; ++c) {
            double expected = 0.0;
            for (int r = 0; r < ROWS; ++r) {
                double p = pred.getUnsafe(r, c);
                double t = targets.getUnsafe(r, c);
                expected -= t * Math.log(p) + (1.0 - t) * Math.log(1.0 - p);
            }
            // the loss is divided by the number of rows, the gradient is not
            assertEquals(expected / ROWS, captured[0].getUnsafe(0, c), 1e-4);
        }
    }

    @Test
    void extremePredictionsStayFinite() {
        MatrixF pred = Matrices.createF(2, 1);
        pred.setUnsafe(0, 0, 0.0f);
        pred.setUnsafe(1, 0, 1.0f);
        MatrixF targets = Matrices.createF(2, 1);
        targets.setUnsafe(0, 0, 1.0f);
        targets.setUnsafe(1, 0, 0.0f);

        MatrixF[] captured = new MatrixF[1];
        BinaryCrossEntropyLoss loss = new BinaryCrossEntropyLoss();
        loss.setMode(NetworkMode.TRAIN);
        loss.registerBatchExpectedValuesProvider(n -> targets);
        loss.registerLossCallback(l -> captured[0] = l);
        MatrixF gradients = loss.forward(pred);

        // the clamp to Float.MIN_NORMAL keeps these large but finite
        assertTrue(Float.isFinite(gradients.getUnsafe(0, 0)), "gradient at p = 0 must stay finite");
        assertTrue(Float.isFinite(gradients.getUnsafe(1, 0)), "gradient at p = 1 must stay finite");
        assertTrue(gradients.getUnsafe(0, 0) < 0.0f, "p = 0 with target 1 must push up");
        assertTrue(gradients.getUnsafe(1, 0) > 0.0f, "p = 1 with target 0 must push down");
        assertTrue(Float.isFinite(captured[0].getUnsafe(0, 0)), "loss must stay finite");
    }

    @Test
    void forwardReturnsNullWithoutExpectedValues() {
        BinaryCrossEntropyLoss loss = new BinaryCrossEntropyLoss();
        loss.setMode(NetworkMode.TRAIN);
        assertNull(loss.forward(probabilities()));
    }

    @Test
    void isNotFusedSoTheNetworkSkipsItDuringInference() {
        assertFalse(new BinaryCrossEntropyLoss().producesPredictionInInferMode());
    }

    private static MatrixF forward(MatrixF pred, MatrixF targets) {
        BinaryCrossEntropyLoss loss = new BinaryCrossEntropyLoss();
        loss.setMode(NetworkMode.TRAIN);
        loss.registerBatchExpectedValuesProvider(n -> targets);
        return loss.forward(pred);
    }

    /** Values well inside (0, 1), so the reference arithmetic stays exact. */
    private static MatrixF probabilities() {
        return Matrices.randomUniformF(ROWS, COLS, 0.15f, 0.85f, 91L);
    }

    private static MatrixF alternatingTargets() {
        MatrixF targets = Matrices.createF(ROWS, COLS);
        for (int c = 0; c < COLS; ++c) {
            for (int r = 0; r < ROWS; ++r) {
                targets.setUnsafe(r, c, (r + c) % 2 == 0 ? 1.0f : 0.0f);
            }
        }
        return targets;
    }
}
