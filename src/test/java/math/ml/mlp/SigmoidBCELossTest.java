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
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

class SigmoidBCELossTest {

    @Test
    void forwardReturnsTheFusedGradient() {
        int rows = 8;
        int cols = 4;
        MatrixF logits = input(rows, cols, 41L);
        MatrixF targets = alternatingTargets(rows, cols);

        MatrixF gradients = trainForward(logits, targets);
        for (int c = 0; c < cols; ++c) {
            for (int r = 0; r < rows; ++r) {
                double p = sigmoid(logits.getUnsafe(r, c));
                assertEquals(p - targets.getUnsafe(r, c), gradients.getUnsafe(r, c), 1e-6);
            }
        }
    }

    @Test
    void saturatedLogitsKeepTheirGradient() {
        // this is what Sigmoid -> BinaryCrossEntropyLoss loses: sigmoid(60) rounds
        // to 1.0f, its derivative becomes 0, and the product underflows to 0
        MatrixF logits = Matrices.createF(2, 1);
        logits.setUnsafe(0, 0, 60.0f);
        logits.setUnsafe(1, 0, -60.0f);
        MatrixF targets = Matrices.createF(2, 1);
        targets.setUnsafe(0, 0, 0.0f);
        targets.setUnsafe(1, 0, 1.0f);

        MatrixF gradients = trainForward(logits, targets);
        assertEquals(1.0f, gradients.getUnsafe(0, 0), 1e-6f, "logit +60 with target 0 must push down");
        assertEquals(-1.0f, gradients.getUnsafe(1, 0), 1e-6f, "logit -60 with target 1 must push up");
    }

    @Test
    void separateSigmoidAndBceLoseTheGradientWhereTheFusedLossDoesNot() {
        // pins the defect the fused loss exists to avoid
        MatrixF logits = Matrices.createF(1, 1);
        logits.setUnsafe(0, 0, 60.0f);
        MatrixF targets = Matrices.createF(1, 1);

        Sigmoid sigmoid = new Sigmoid();
        sigmoid.setMode(NetworkMode.TRAIN);
        BinaryCrossEntropyLoss bce = new BinaryCrossEntropyLoss();
        bce.setMode(NetworkMode.TRAIN);
        bce.setExpectedValues(targets);

        MatrixF separate = sigmoid.backward(bce.forward(sigmoid.forward(logits.copy())), 0.0f);
        assertEquals(0.0f, separate.getUnsafe(0, 0), 0.0f, "the unfused pair is expected to lose this gradient");
        assertEquals(1.0f, trainForward(logits, targets).getUnsafe(0, 0), 1e-6f);
    }

    @Test
    void inferenceReturnsProbabilitiesInsteadOfGradients() {
        MatrixF logits = input(6, 3, 42L);
        SigmoidBCELoss loss = new SigmoidBCELoss();
        loss.setMode(NetworkMode.INFER);
        MatrixF out = loss.forward(logits);
        for (int c = 0; c < logits.numColumns(); ++c) {
            for (int r = 0; r < logits.numRows(); ++r) {
                assertEquals(sigmoid(logits.getUnsafe(r, c)), out.getUnsafe(r, c), 1e-6);
            }
        }
    }

    @Test
    void backwardHandsBackTheGradientComputedInForward() {
        MatrixF logits = input(5, 2, 43L);
        MatrixF targets = alternatingTargets(5, 2);
        SigmoidBCELoss loss = new SigmoidBCELoss();
        loss.setMode(NetworkMode.TRAIN);
        loss.setExpectedValues(targets);

        MatrixF fromForward = loss.forward(logits);
        MatrixF fromBackward = loss.backward(null, 0.0f);
        assertNotNull(fromBackward);
        for (int c = 0; c < 2; ++c) {
            for (int r = 0; r < 5; ++r) {
                assertEquals(fromForward.getUnsafe(r, c), fromBackward.getUnsafe(r, c), 0.0f);
            }
        }
    }

    @Test
    void reportedLossIsFiniteAtSaturationAndMatchesTheReference() {
        MatrixF logits = Matrices.createF(3, 1);
        logits.setUnsafe(0, 0, 80.0f);
        logits.setUnsafe(1, 0, -80.0f);
        logits.setUnsafe(2, 0, 0.5f);
        MatrixF targets = Matrices.createF(3, 1);
        targets.setUnsafe(0, 0, 1.0f);
        targets.setUnsafe(1, 0, 1.0f);
        targets.setUnsafe(2, 0, 0.0f);

        MatrixF[] captured = new MatrixF[1];
        SigmoidBCELoss loss = new SigmoidBCELoss();
        loss.setMode(NetworkMode.TRAIN);
        loss.setExpectedValues(targets);
        loss.registerLossCallback(l -> captured[0] = l);
        loss.forward(logits);

        assertNotNull(captured[0]);
        float reported = captured[0].getUnsafe(0, 0);
        assertTrue(Float.isFinite(reported), "loss must stay finite at |x| = 80");
        // reference: sum over rows of max(x,0) - x*t + log1p(exp(-|x|))
        double expected = 0.0;
        for (int r = 0; r < 3; ++r) {
            double x = logits.getUnsafe(r, 0);
            double t = targets.getUnsafe(r, 0);
            expected += Math.max(x, 0.0) - x * t + Math.log1p(Math.exp(-Math.abs(x)));
        }
        assertEquals(expected, reported, 1e-3);
    }

    @Test
    void producesPredictionInInferModeSoTheNetworkKeepsItInTheChain() {
        assertTrue(new SigmoidBCELoss().producesPredictionInInferMode());
        assertTrue(new SoftmaxCrossEntropyLoss().producesPredictionInInferMode());
        assertTrue(!new BinaryCrossEntropyLoss().producesPredictionInInferMode());
        assertTrue(!new CrossEntropyLoss().producesPredictionInInferMode());
    }

    private static MatrixF trainForward(MatrixF logits, MatrixF targets) {
        SigmoidBCELoss loss = new SigmoidBCELoss();
        loss.setMode(NetworkMode.TRAIN);
        loss.setExpectedValues(targets);
        return loss.forward(logits);
    }

    private static MatrixF alternatingTargets(int rows, int cols) {
        MatrixF targets = Matrices.createF(rows, cols);
        for (int c = 0; c < cols; ++c) {
            for (int r = 0; r < rows; ++r) {
                targets.setUnsafe(r, c, (r + c) % 2 == 0 ? 1.0f : 0.0f);
            }
        }
        return targets;
    }

    private static double sigmoid(float x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
}
