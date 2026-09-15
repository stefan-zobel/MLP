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
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

class NetworkPersistenceTest {

    @TempDir
    Path dir;

    @Test
    void inferenceDoesNotPersistAnything() {
        CountingLayer probe = new CountingLayer("probe");
        Net net = new Net();
        net.add(probe);

        net.infer(input(4, 3, 101L));
        net.infer(input(4, 3, 102L));

        assertEquals(0, probe.written, "infer() must be free of persistence side effects");
        assertFalse(Files.exists(dir.resolve("m.zip")));
    }

    @Test
    void storingReachesEveryLayerExactlyOnce() {
        CountingLayer first = new CountingLayer("first");
        CountingLayer second = new CountingLayer("second");
        Net net = new Net();
        net.add(first);
        net.add(second);

        net.storeParameters(dir.resolve("m.zip"));

        assertEquals(1, first.written);
        assertEquals(1, second.written);
    }

    @Test
    void aLayerReachableTwiceIsRefused() {
        CountingLayer inBranch = new CountingLayer("shared");
        Net net = new Net();
        net.add(new ResidualBranch(inBranch));
        net.add(new ParallelBranches(List.of(new CountingLayer("other"), inBranch)));

        // once through the residual branch, once through the parallel branch, and the second
        // one is an entry that exists already
        assertTrue(assertThrows(IllegalStateException.class, () -> net.storeParameters(dir.resolve("m.zip")))
                .getMessage().contains("shared/probe"));
        assertFalse(Files.exists(dir.resolve("m.zip")), "a refused write must leave nothing behind");
    }

    @Test
    void twoLayersOfTheSameNameAreRefused() {
        Net net = new Net();
        net.add(new Hidden(4, 3, "twice", 11L));
        net.add(new Hidden(3, 2, "twice", 12L));

        assertTrue(assertThrows(IllegalStateException.class, () -> net.storeParameters(dir.resolve("m.zip")))
                .getMessage().contains("twice/weights"));
    }

    @Test
    void aTrainableLayerWithoutANameStopsTheWholeBundle() {
        Net net = new Net();
        net.add(new Hidden(4, 3, "named", 11L));
        // trained by the optimizer, and silently absent from the bundle until this threw
        net.add(new LayerNorm(3));
        net.add(new Hidden(3, 2, "also_named", 12L));

        assertTrue(assertThrows(IllegalStateException.class, () -> net.storeParameters(dir.resolve("m.zip")))
                .getMessage().contains("LayerNorm"));
        assertFalse(Files.exists(dir.resolve("m.zip")), "a model missing weights must not reach the disk");
    }

    @Test
    void aNamedNormalizationLayerIsInTheBundle() {
        Path file = dir.resolve("m.zip");
        Net net = new Net();
        net.add(new Hidden(4, 3, "one", 11L));
        net.add(new LayerNorm(3, "one_ln"));
        net.storeParameters(file);

        Net read = new Net();
        read.add(new Hidden(4, 3, "one", 99L));
        read.add(new LayerNorm(3, "one_ln"));
        read.loadParameters(file);

        assertSameParameters(net, read);
    }

    @Test
    void aLayerThatLeavesOneOfItsMatricesOutIsRefused() {
        Net net = new Net();
        net.add(new Hidden(4, 3, "complete", 11L));
        net.add(new ForgetfulLayer("forgetful"));

        String message = assertThrows(IllegalStateException.class,
                () -> net.storeParameters(dir.resolve("m.zip"))).getMessage();
        assertTrue(message.contains("ForgetfulLayer"), message);
        assertTrue(message.contains("2") && message.contains("1"), message);
        assertFalse(Files.exists(dir.resolve("m.zip")), "a model missing weights must not reach the disk");
    }

    @Test
    void aLayerWithMoreEntriesThanParametersIsFine() {
        Path file = dir.resolve("m.zip");
        Net net = new Net();
        // two trainable matrices, four entries: the running statistics are not trainable and
        // are exactly the slack the check tolerates
        net.add(new BatchNorm(3, "norm"));
        net.storeParameters(file);

        Net read = new Net();
        read.add(new BatchNorm(3, "norm"));
        assertEquals(0, read.loadParameters(file));
    }

    @Test
    void aNetworkOfLayersWithoutParametersStillStores() {
        Path file = dir.resolve("m.zip");
        Net net = new Net();
        net.add(new Relu());
        net.add(new Relu());

        net.storeParameters(file);

        assertTrue(Files.exists(file));
    }

    @Test
    void whatTheNetworkWroteItReadsBack() {
        Path file = dir.resolve("m.zip");
        Net written = twoLayers(13L);
        written.storeParameters(file);

        Net read = twoLayers(99L);
        assertEquals(0, read.loadParameters(file));

        assertSameParameters(written, read);
    }

    @Test
    void aBundleOfAnotherArchitectureIsRefused() {
        Path file = dir.resolve("m.zip");
        twoLayers(13L).storeParameters(file);

        Net narrower = new Net();
        narrower.add(new Hidden(4, 3, "one", 21L));
        assertTrue(assertThrows(IllegalStateException.class, () -> narrower.loadParameters(file)).getMessage()
                .contains("two/"), "the entries of the layer this network does not have must be named");
    }

    @Test
    void aBundleMissingALayerIsRefused() {
        Path file = dir.resolve("m.zip");
        Net one = new Net();
        one.add(new Hidden(4, 3, "one", 21L));
        one.storeParameters(file);

        assertTrue(assertThrows(IllegalStateException.class, () -> twoLayers(13L).loadParameters(file)).getMessage()
                .contains("two/weights"));
    }

    @Test
    void theStepTheNetworkWasAtComesBackWithIt() {
        Path file = dir.resolve("m.zip");
        Net trained = trainableNet(31L);
        trainBatches(trained, 0, 7);
        trained.storeParameters(file);

        Net fresh = trainableNet(32L);
        assertEquals(7, fresh.loadParameters(file));
    }

    @Test
    void theStepKeepsCountingAcrossAResume() {
        Path first = dir.resolve("first.zip");
        Net trained = trainableNet(31L);
        trainBatches(trained, 0, 7);
        trained.storeParameters(first);

        Net resumed = trainableNet(32L);
        resumed.loadParameters(first);
        trainBatches(resumed, 7, 10);
        Path second = dir.resolve("second.zip");
        resumed.storeParameters(second);

        // without this the second checkpoint would count only the batches since the resume,
        // and a run continued from it would replay the schedule it already went through
        assertEquals(10, trainableNet(33L).loadParameters(second));
    }

    @Test
    void resumeTurnsTheStepIntoTheEpochToContinueAt() {
        Path file = dir.resolve("m.zip");
        Net trained = trainableNet(41L);
        trainBatches(trained, 0, 10);
        trained.storeParameters(file);

        // ten batches at four to the epoch: two epochs are done, the third is half trained
        // and is trained again, which is the only whole number available here
        assertEquals(2, trainableNet(42L).resume(file, 4));
        assertEquals(10, trainableNet(43L).resume(file, 1));
    }

    @Test
    void resumeRejectsAnEpochOfNoBatches() {
        Path file = dir.resolve("m.zip");
        trainableNet(41L).storeParameters(file);

        assertThrows(IllegalArgumentException.class, () -> trainableNet(42L).resume(file, 0));
    }

    private static void trainBatches(Net net, int from, int to) {
        for (int b = from; b < to; ++b) {
            net.train(input(4, 5, 400L + b), input(2, 5, 500L + b));
        }
    }

    private static Net twoLayers(long seed) {
        Net net = new Net();
        net.add(new Hidden(4, 3, "one", seed));
        net.add(new Hidden(3, 2, "two", seed + 1L));
        return net;
    }

    private static Net trainableNet(long seed) {
        Net net = new Net();
        net.add(new Hidden(4, 2, "only", seed));
        net.add(new SoftmaxCrossEntropyLoss());
        net.optimizer(new Adam(0.01f));
        return net;
    }

    private static void assertSameParameters(Net expected, Net actual) {
        List<Parameter> a = expected.every();
        List<Parameter> b = actual.every();
        assertEquals(a.size(), b.size());
        for (int i = 0; i < a.size(); ++i) {
            assertArrayEquals(a.get(i).value().getArrayUnsafe(), b.get(i).value().getArrayUnsafe(),
                    a.get(i).name() + " " + i);
        }
    }

    private static final class Net extends AbstractNetwork {

        List<Parameter> every() {
            return layers.stream().flatMap(l -> l.parameters().stream()).toList();
        }
    }

    // declares two trainable matrices and writes one, which is what no structural check
    // caught until the count below
    private static final class ForgetfulLayer extends AbstractLayer {

        private final String name;
        private final Parameter kept;
        private final Parameter dropped;

        ForgetfulLayer(String name) {
            this.name = name;
            this.kept = new Parameter("kept", Matrices.createF(2, 2), true);
            this.dropped = new Parameter("dropped", Matrices.createF(2, 2), true);
        }

        @Override
        public List<Parameter> parameters() {
            return List.of(kept, dropped);
        }

        @Override
        public MatrixF forward(MatrixF in) {
            return in;
        }

        @Override
        public MatrixF backward(MatrixF grads) {
            return grads;
        }

        @Override
        public void writeParameters(ParameterSink sink) throws IOException {
            ParameterStore.write(sink, name + "/kept", kept.value());
        }
    }

    private static final class CountingLayer extends AbstractLayer {

        private final String name;
        int written;

        CountingLayer(String name) {
            this.name = name;
        }

        @Override
        public MatrixF forward(MatrixF in) {
            return in;
        }

        @Override
        public MatrixF backward(MatrixF grads) {
            return grads;
        }

        @Override
        public void writeParameters(ParameterSink sink) throws IOException {
            ++written;
            try (OutputStream os = sink.open(name + "/probe")) {
                os.write(1);
            }
        }
    }
}
