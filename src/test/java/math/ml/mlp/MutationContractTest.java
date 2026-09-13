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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

class MutationContractTest {

    @Test
    void dropoutDeclaresBothMutations() {
        Dropout dropout = new Dropout(0.3f, 1L);
        assertTrue(dropout.mutatesInput());
        assertTrue(dropout.mutatesGradients());
    }

    @Test
    void aDropoutThatDropsNothingMutatesNothing() {
        Dropout dropout = new Dropout(0.0f, 2L);
        assertFalse(dropout.mutatesInput());
        assertFalse(dropout.mutatesGradients());
    }

    @Test
    void everyOtherLayerLeavesItsArgumentsAlone() {
        List<Layer> layers = List.of(new Hidden(4, 4, "mc1"), new Relu(), new Gelu(), new Sigmoid(), new Softmax(),
                new BatchNorm(4), new LayerNorm(4), new VAEReparamLayer(2, 1.0f), new CrossEntropyLoss(),
                new SoftmaxCrossEntropyLoss(), new BinaryCrossEntropyLoss(), new SigmoidBCELoss());
        for (Layer layer : layers) {
            String name = layer.getClass().getSimpleName();
            assertFalse(layer.mutatesInput(), name + " declares that it mutates its input");
            assertFalse(layer.mutatesGradients(), name + " declares that it mutates its gradients");
        }
    }

    @Test
    void aCompositeAnswersForItsOwnArgumentsNotForItsBranch() {
        // both composites allocate what they return, so a Dropout inside one of them
        // is invisible from the outside -- the copy happens within
        ResidualBranch residual = new ResidualBranch(new Dropout(0.3f, 3L));
        assertFalse(residual.mutatesInput());
        assertFalse(residual.mutatesGradients());

        ParallelBranches parallel = new ParallelBranches(List.of(new Dropout(0.3f, 4L)));
        assertFalse(parallel.mutatesInput());
        assertFalse(parallel.mutatesGradients());
    }

    @Test
    void aBranchWithoutAMutatingLayerSeesTheSharedMatrixItself() {
        RecordingLayer probe = new RecordingLayer();
        ResidualBranch layer = new ResidualBranch(probe);
        layer.setMode(NetworkMode.TRAIN);
        MatrixF x = input(4, 3, 41L);
        layer.forward(x);
        assertSame(x, probe.received, "nothing in this branch mutates, so no copy was needed");
    }

    @Test
    void aBranchWithAMutatingLayerSeesACopy() {
        RecordingLayer probe = new RecordingLayer();
        ResidualBranch layer = new ResidualBranch(probe, new Dropout(0.3f, 5L));
        layer.setMode(NetworkMode.TRAIN);
        MatrixF x = input(4, 3, 42L);
        layer.forward(x);
        assertNotSame(x, probe.received, "the Dropout behind the probe forces a copy");
    }

    @Test
    void trainingProtectsTheCallersMatrixFromALeadingDropout() {
        Net net = new Net();
        net.add(new Dropout(0.5f, 6L));
        net.add(new SigmoidBCELoss());

        MatrixF x = input(4, 3, 43L);
        MatrixF before = x.copy();
        net.train(x, Matrices.randomUniformF(4, 3, 0.0f, 1.0f, 44L));

        for (int c = 0; c < x.numColumns(); ++c) {
            for (int r = 0; r < x.numRows(); ++r) {
                assertEquals(before.getUnsafe(r, c), x.getUnsafe(r, c), "element [" + r + "," + c + "] was masked");
            }
        }
    }

    // The predicates describe arguments. What a layer returns is a separate property, and
    // BatchNorm and LayerNorm must return something freshly allocated: Activation caches
    // the previous layer's output until its own backward, infer() hands the last output to
    // the caller, and ParallelBranches holds every branch output until it stacks them.
    @Test
    void theNormLayersReturnAFreshMatrixEveryTime() {
        assertFreshReturn(new BatchNorm(4), NetworkMode.TRAIN);
        assertFreshReturn(new BatchNorm(4), NetworkMode.INFER);
        assertFreshReturn(new LayerNorm(4), NetworkMode.TRAIN);
        assertFreshReturn(new LayerNorm(4), NetworkMode.INFER);
    }

    private static void assertFreshReturn(Layer layer, NetworkMode mode) {
        layer.setMode(mode);
        MatrixF x = input(4, 3, 61L);
        MatrixF first = layer.forward(x);
        assertNotSame(x, first, "forward returned its own argument");
        MatrixF second = layer.forward(x);
        assertNotSame(first, second, "forward reused the matrix it returned last time");
        if (mode == NetworkMode.TRAIN) {
            MatrixF grads = input(4, 3, 62L);
            MatrixF back = layer.backward(grads);
            assertNotSame(grads, back, "backward returned its own argument");
            assertNotSame(second, back, "backward returned the matrix forward had returned");
        }
    }
    // remembers the matrix it was handed, so a copy can be told from the original
    private static final class RecordingLayer extends AbstractLayer {

        private MatrixF received;

        @Override
        public MatrixF forward(MatrixF in) {
            received = in;
            return in;
        }

        @Override
        public MatrixF backward(MatrixF grads) {
            return grads;
        }
    }

    private static final class Net extends AbstractNetwork {
    }
}
