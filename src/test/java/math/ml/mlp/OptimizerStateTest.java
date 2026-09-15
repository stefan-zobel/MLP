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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

class OptimizerStateTest {

    private static final float RATE = 0.01f;

    @TempDir
    Path dir;

    @Test
    void adamResumesExactlyWhereItStopped() throws IOException {
        List<Parameter> straight = twoParameters();
        Adam whole = new Adam(RATE);
        straight.forEach(whole::add);
        stepThrough(whole, straight, 1, 10);

        List<Parameter> before = twoParameters();
        AbstractOptimizer first = new Adam(RATE);
        before.forEach(first::add);
        stepThrough(first, before, 1, 5);
        MemoryBundle bundle = new MemoryBundle();
        first.writeTo(bundle.sink());

        List<Parameter> after = carriedOver(before);
        AbstractOptimizer second = new Adam(RATE);
        after.forEach(second::add);
        second.readFrom(bundle.source(), 5);
        assertEquals(5, second.steps());
        stepThrough(second, after, 6, 10);

        assertSameValues(straight, after);
    }

    @Test
    void sgdWithMomentumResumesExactlyWhereItStopped() throws IOException {
        List<Parameter> straight = twoParameters();
        Sgd whole = new Sgd(RATE, 0.9f);
        straight.forEach(whole::add);
        stepThrough(whole, straight, 1, 10);

        List<Parameter> before = twoParameters();
        AbstractOptimizer first = new Sgd(RATE, 0.9f);
        before.forEach(first::add);
        stepThrough(first, before, 1, 5);
        MemoryBundle bundle = new MemoryBundle();
        first.writeTo(bundle.sink());

        List<Parameter> after = carriedOver(before);
        AbstractOptimizer second = new Sgd(RATE, 0.9f);
        after.forEach(second::add);
        second.readFrom(bundle.source(), 5);
        stepThrough(second, after, 6, 10);

        assertSameValues(straight, after);
    }

    @Test
    void plainSgdCarriesNothingButTheStepCounter() throws IOException {
        List<Integer> asked = new ArrayList<>();
        List<Parameter> before = twoParameters();
        AbstractOptimizer first = new Sgd(recording(asked));
        before.forEach(first::add);
        stepThrough(first, before, 1, 5);
        MemoryBundle bundle = new MemoryBundle();
        first.writeTo(bundle.sink());

        asked.clear();
        List<Parameter> after = carriedOver(before);
        AbstractOptimizer second = new Sgd(recording(asked));
        after.forEach(second::add);
        second.readFrom(bundle.source(), 5);
        stepThrough(second, after, 6, 10);

        assertEquals(List.of(6, 7, 8, 9, 10), asked);
    }

    @Test
    void theResumedRunContinuesTheScheduleInsteadOfRepeatingIt() throws IOException {
        List<Integer> asked = new ArrayList<>();
        List<Parameter> before = twoParameters();
        AbstractOptimizer first = new Adam(recording(asked), 0.0f);
        before.forEach(first::add);
        stepThrough(first, before, 1, 5);
        assertEquals(List.of(1, 2, 3, 4, 5), asked);
        MemoryBundle bundle = new MemoryBundle();
        first.writeTo(bundle.sink());

        asked.clear();
        AbstractOptimizer second = new Adam(recording(asked), 0.0f);
        carriedOver(before).forEach(second::add);
        second.readFrom(bundle.source(), 5);
        second.step();

        assertEquals(List.of(6), asked);
    }

    @Test
    void theStepComesFromTheBundleAndNotFromTheEntry() throws IOException {
        List<Parameter> params = twoParameters();
        AbstractOptimizer first = new Adam(RATE);
        params.forEach(first::add);
        stepThrough(first, params, 1, 5);
        MemoryBundle bundle = new MemoryBundle();
        first.writeTo(bundle.sink());

        AbstractOptimizer second = new Adam(RATE);
        carriedOver(params).forEach(second::add);
        // the entry holds no step of its own, so whatever the manifest says is what counts
        second.readFrom(bundle.source(), 1128);

        assertEquals(1128, second.steps());
    }

    @Test
    void theClippedStepCountSurvives() throws IOException {
        List<Parameter> before = twoParameters();
        AbstractOptimizer first = new Adam(RATE);
        first.clipGradientNorm(1e-6f);
        before.forEach(first::add);
        stepThrough(first, before, 1, 5);
        assertEquals(5, first.clippedSteps());
        MemoryBundle bundle = new MemoryBundle();
        first.writeTo(bundle.sink());

        AbstractOptimizer second = new Adam(RATE);
        carriedOver(before).forEach(second::add);
        second.readFrom(bundle.source(), 5);

        assertEquals(5, second.clippedSteps());
    }

    @Test
    void stateWrittenByOneKindOfOptimizerIsRefusedByAnother() throws IOException {
        MemoryBundle bundle = fiveStepsOf(new Adam(RATE));

        AbstractOptimizer sgd = new Sgd(RATE, 0.9f);
        twoParameters().forEach(sgd::add);
        assertTrue(assertThrows(IllegalStateException.class, () -> sgd.readFrom(bundle.source(), 5)).getMessage()
                .contains("Adam"));
    }

    @Test
    void aDifferentParameterCountIsRefused() throws IOException {
        MemoryBundle bundle = fiveStepsOf(new Adam(RATE));

        AbstractOptimizer narrow = new Adam(RATE);
        narrow.add(twoParameters().get(0));
        assertThrows(IllegalStateException.class, () -> narrow.readFrom(bundle.source(), 5));
    }

    @Test
    void aDifferentParameterShapeIsRefused() throws IOException {
        MemoryBundle bundle = fiveStepsOf(new Adam(RATE));

        AbstractOptimizer wider = new Adam(RATE);
        wider.add(new Parameter("w", Matrices.createF(5, 3), true));
        wider.add(new Parameter("b", Matrices.createF(4, 1), false));
        assertThrows(IllegalStateException.class, () -> wider.readFrom(bundle.source(), 5));
    }

    @Test
    void aChangedHyperparameterIsRefused() throws IOException {
        MemoryBundle bundle = fiveStepsOf(new Adam(RATE));

        AbstractOptimizer other = new Adam(LearningRateSchedule.constant(RATE), 0.0f, 0.8f, 0.999f, 1e-8f);
        twoParameters().forEach(other::add);
        assertTrue(assertThrows(IllegalStateException.class, () -> other.readFrom(bundle.source(), 5)).getMessage()
                .contains("beta1"));
    }

    @Test
    void aChangedMomentumIsRefused() throws IOException {
        MemoryBundle bundle = fiveStepsOf(new Sgd(RATE, 0.9f));

        AbstractOptimizer other = new Sgd(RATE, 0.5f);
        twoParameters().forEach(other::add);
        assertTrue(assertThrows(IllegalStateException.class, () -> other.readFrom(bundle.source(), 5)).getMessage()
                .contains("momentum"));
    }

    @Test
    void aBundleWithoutAnOptimizerEntryIsReported() {
        AbstractOptimizer adam = new Adam(RATE);
        twoParameters().forEach(adam::add);

        assertTrue(assertThrows(IllegalStateException.class, () -> adam.readFrom(new MemoryBundle().source(), 5)).getMessage()
                .contains("optimizer"));
    }

    @Test
    void theNetworkWritesTheOptimizerAlongWithTheLayers() throws IOException {
        Net net = new Net();
        net.add(new Hidden(4, 3, "optimizer_state_test_layer", 41L));
        net.optimizer(new Adam(RATE));

        Path file = dir.resolve("m.zip");
        net.storeParameters(file);

        try (ModelBundle.Reader in = ModelBundle.read(file)) {
            in.open("optimizer").close();
            in.open("optimizer_state_test_layer/weights").close();
            in.open("optimizer_state_test_layer/biases").close();
            in.requireFullyRead();
        }
    }

    private static MemoryBundle fiveStepsOf(AbstractOptimizer optimizer) throws IOException {
        List<Parameter> params = twoParameters();
        params.forEach(optimizer::add);
        stepThrough(optimizer, params, 1, 5);
        MemoryBundle bundle = new MemoryBundle();
        optimizer.writeTo(bundle.sink());
        return bundle;
    }

    private static List<Parameter> twoParameters() {
        return List.of(new Parameter("w", Matrices.randomUniformF(4, 3, -1.0f, 1.0f, 61L), true),
                new Parameter("b", Matrices.randomUniformF(4, 1, -1.0f, 1.0f, 62L), false));
    }

    // what a layer does when it reads its weights back, without any of the layers
    private static List<Parameter> carriedOver(List<Parameter> from) {
        List<Parameter> fresh = twoParameters();
        for (int i = 0; i < fresh.size(); ++i) {
            fresh.get(i).value().setInplace(from.get(i).value());
        }
        return fresh;
    }

    private static void stepThrough(Optimizer optimizer, List<Parameter> params, int from, int to) {
        for (int t = from; t <= to; ++t) {
            for (int i = 0; i < params.size(); ++i) {
                MatrixF g = params.get(i).grad();
                g.setInplace(Matrices.randomUniformF(g.numRows(), g.numColumns(), -0.5f, 0.5f, 1000L * t + i));
            }
            optimizer.step();
        }
    }

    private static LearningRateSchedule recording(List<Integer> asked) {
        return step -> {
            asked.add(step);
            return RATE;
        };
    }

    private static void assertSameValues(List<Parameter> expected, List<Parameter> actual) {
        for (int i = 0; i < expected.size(); ++i) {
            assertArrayEquals(expected.get(i).value().getArrayUnsafe(), actual.get(i).value().getArrayUnsafe(),
                    "parameter " + i);
        }
    }

    private static final class Net extends AbstractNetwork {
    }
}
