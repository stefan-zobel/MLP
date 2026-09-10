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
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

class NetworkTrainingTest {

    @Test
    void everyBatchIsScoredAgainstTheTargetsItWasGiven() {
        // the same input trained twice used to shift the loss onto the next
        // batch's targets, because the loss counted batches on its own
        RecordingLoss loss = new RecordingLoss();
        Net net = netWith(loss);

        MatrixF x = input(4, 2, 401L);
        MatrixF first = input(4, 2, 402L);
        MatrixF second = input(4, 2, 403L);

        net.train(x, first, 0.01f);
        net.train(x, second, 0.01f);
        net.train(x, first, 0.01f);

        assertEquals(List.of(first, second, first), loss.seen);
    }

    @Test
    void trainingRefusesNullTargets() {
        Net net = netWith(new RecordingLoss());
        assertThrows(IllegalArgumentException.class, () -> net.train(input(4, 2, 404L), null, 0.01f));
    }

    @Test
    void trainingRefusesANetworkWithoutATrailingLoss() {
        Net net = new Net();
        net.add(new Relu());
        net.add(new Relu());
        assertThrows(IllegalStateException.class,
                () -> net.train(input(4, 2, 405L), input(4, 2, 406L), 0.01f));
    }

    @Test
    void trainingRefusesASingleLayerNetwork() {
        Net net = new Net();
        net.add(new RecordingLoss());
        assertThrows(IllegalStateException.class,
                () -> net.train(input(4, 2, 407L), input(4, 2, 408L), 0.01f));
    }

    @Test
    void trainReturnsTheNetworkSoCallsCanBeChained() {
        Net net = netWith(new RecordingLoss());
        assertSame(net, net.train(input(4, 2, 409L), input(4, 2, 410L), 0.01f));
    }

    @Test
    void thesameSeedTrainsTwoNetworksIdentically() {
        // the payoff of seeding: everything random in a run is derived from one long
        assertEquals(trainingLosses(11L), trainingLosses(11L));
    }

    @Test
    void aDifferentSeedTrainsDifferently() {
        assertNotEquals(trainingLosses(11L), trainingLosses(12L));
    }

    /** Trains a Hidden + Dropout net for a few batches and returns the loss per batch. */
    private static List<Float> trainingLosses(long baseSeed) {
        SplittableRandom seeds = new SplittableRandom(baseSeed);
        List<Float> losses = new ArrayList<>();

        SigmoidBCELoss loss = new SigmoidBCELoss();
        loss.registerLossCallback(l -> losses.add(Matrices.colsAverage(l).toScalar()));

        Net net = new Net();
        net.add(new Hidden(8, 5, "a", seeds.nextLong()));
        net.add(new Dropout(0.3f, seeds.nextLong()));
        net.add(new Relu());
        net.add(new Hidden(5, 3, "b", seeds.nextLong()));
        net.add(loss);

        MatrixF targets = Matrices.randomUniformF(3, 6, 0.0f, 1.0f, 511L);
        for (int batch = 0; batch < 5; ++batch) {
            net.train(input(8, 6, 512L), targets, 0.05f);
        }
        return losses;
    }

    private static Net netWith(Loss loss) {
        Net net = new Net();
        net.add(new Relu());
        net.add(loss);
        return net;
    }

    private static final class Net extends AbstractNetwork {
    }

    /** Records the targets it was handed, in order. */
    private static final class RecordingLoss extends AbstractLoss {

        final List<MatrixF> seen = new ArrayList<>();

        @Override
        public MatrixF forward(MatrixF prediction) {
            seen.add(getExpectation());
            return prediction;
        }
    }
}
