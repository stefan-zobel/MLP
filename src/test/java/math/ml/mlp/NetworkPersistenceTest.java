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

import org.junit.jupiter.api.Test;

import net.jamu.matrix.MatrixF;

class NetworkPersistenceTest {

    @Test
    void inferenceDoesNotPersistAnything() {
        CountingLayer probe = new CountingLayer();
        Net net = new Net();
        net.add(probe);

        net.infer(input(4, 3, 101L));
        net.infer(input(4, 3, 102L));

        assertEquals(0, probe.stored, "infer() must be free of persistence side effects");
    }

    @Test
    void storeParametersReachesEveryLayerExactlyOnce() {
        CountingLayer first = new CountingLayer();
        CountingLayer second = new CountingLayer();
        Net net = new Net();
        net.add(first);
        net.add(second);

        net.storeParameters();

        assertEquals(1, first.stored);
        assertEquals(1, second.stored);
    }

    @Test
    void storeParametersReachesLayersNestedInComposites() {
        CountingLayer inBranch = new CountingLayer();
        Net net = new Net();
        net.add(new ResidualBranch(inBranch));
        net.add(new ParallelBranches(java.util.List.of(new CountingLayer(), inBranch)));

        net.storeParameters();

        // once through the residual branch, once through the parallel branch
        assertEquals(2, inBranch.stored);
    }

    private static final class Net extends AbstractNetwork {
    }

    private static final class CountingLayer extends AbstractLayer {
        int stored;

        @Override
        public MatrixF forward(MatrixF in) {
            return in;
        }

        @Override
        public MatrixF backward(MatrixF grads) {
            return grads;
        }

        @Override
        public void storeParameters() {
            ++stored;
        }
    }
}
