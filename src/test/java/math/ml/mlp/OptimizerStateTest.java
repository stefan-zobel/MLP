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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

class OptimizerStateTest {

    private static final String NAME = "optimizer_state_test";
    private static final Path WRITTEN = Paths.get("./checkpoints/o_" + NAME);
    private static final Path PROMOTED = Paths.get("./data/o_" + NAME);
    private static final Path PARTIAL = Paths.get("./checkpoints/o_" + NAME + ".tmp");
    private static final float RATE = 0.01f;

    @AfterEach
    void removeWhatTheTestWrote() throws IOException {
        Files.deleteIfExists(WRITTEN);
        Files.deleteIfExists(PROMOTED);
        Files.deleteIfExists(PARTIAL);
    }

    @Test
    void adamResumesExactlyWhereItStopped() throws IOException {
        List<Parameter> straight = twoParameters();
        Adam whole = new Adam(RATE);
        straight.forEach(whole::add);
        stepThrough(whole, straight, 1, 10);

        List<Parameter> before = twoParameters();
        AbstractOptimizer first = new Adam(RATE).persistAs(NAME, true);
        before.forEach(first::add);
        stepThrough(first, before, 1, 5);
        first.storeState();
        promote();

        List<Parameter> after = carriedOver(before);
        AbstractOptimizer second = new Adam(RATE).persistAs(NAME, false);
        after.forEach(second::add);
        assertEquals(5, second.loadState());
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
        AbstractOptimizer first = new Sgd(RATE, 0.9f).persistAs(NAME, true);
        before.forEach(first::add);
        stepThrough(first, before, 1, 5);
        first.storeState();
        promote();

        List<Parameter> after = carriedOver(before);
        AbstractOptimizer second = new Sgd(RATE, 0.9f).persistAs(NAME, false);
        after.forEach(second::add);
        second.loadState();
        stepThrough(second, after, 6, 10);

        assertSameValues(straight, after);
    }

    @Test
    void plainSgdCarriesNothingButTheStepCounter() throws IOException {
        List<Integer> asked = new ArrayList<>();
        List<Parameter> before = twoParameters();
        AbstractOptimizer first = new Sgd(recording(asked)).persistAs(NAME, true);
        before.forEach(first::add);
        stepThrough(first, before, 1, 5);
        first.storeState();
        promote();

        asked.clear();
        List<Parameter> after = carriedOver(before);
        AbstractOptimizer second = new Sgd(recording(asked)).persistAs(NAME, false);
        after.forEach(second::add);
        assertEquals(5, second.loadState());
        stepThrough(second, after, 6, 10);

        assertEquals(List.of(6, 7, 8, 9, 10), asked);
    }

    @Test
    void theResumedRunContinuesTheScheduleInsteadOfRepeatingIt() throws IOException {
        List<Integer> asked = new ArrayList<>();
        List<Parameter> before = twoParameters();
        AbstractOptimizer first = new Adam(recording(asked), 0.0f).persistAs(NAME, true);
        before.forEach(first::add);
        stepThrough(first, before, 1, 5);
        assertEquals(List.of(1, 2, 3, 4, 5), asked);
        first.storeState();
        promote();

        asked.clear();
        AbstractOptimizer second = new Adam(recording(asked), 0.0f).persistAs(NAME, false);
        carriedOver(before).forEach(second::add);
        second.loadState();
        second.step();

        assertEquals(List.of(6), asked);
    }

    @Test
    void theClippedStepCountSurvives() throws IOException {
        List<Parameter> before = twoParameters();
        AbstractOptimizer first = new Adam(RATE).persistAs(NAME, true);
        first.clipGradientNorm(1e-6f);
        before.forEach(first::add);
        stepThrough(first, before, 1, 5);
        assertEquals(5, first.clippedSteps());
        first.storeState();
        promote();

        AbstractOptimizer second = new Adam(RATE).persistAs(NAME, false);
        carriedOver(before).forEach(second::add);
        second.loadState();

        assertEquals(5, second.clippedSteps());
    }

    @Test
    void stateWrittenByOneKindOfOptimizerIsRefusedByAnother() throws IOException {
        storeFive(new Adam(RATE).persistAs(NAME, true));

        AbstractOptimizer sgd = new Sgd(RATE, 0.9f).persistAs(NAME, false);
        twoParameters().forEach(sgd::add);
        assertTrue(assertThrows(IllegalStateException.class, sgd::loadState).getMessage().contains("Adam"));
    }

    @Test
    void aDifferentParameterCountIsRefused() throws IOException {
        storeFive(new Adam(RATE).persistAs(NAME, true));

        AbstractOptimizer narrow = new Adam(RATE).persistAs(NAME, false);
        narrow.add(twoParameters().get(0));
        assertThrows(IllegalStateException.class, narrow::loadState);
    }

    @Test
    void aDifferentParameterShapeIsRefused() throws IOException {
        storeFive(new Adam(RATE).persistAs(NAME, true));

        AbstractOptimizer wider = new Adam(RATE).persistAs(NAME, false);
        wider.add(new Parameter("w", Matrices.createF(5, 3), true));
        wider.add(new Parameter("b", Matrices.createF(4, 1), false));
        assertThrows(IllegalStateException.class, wider::loadState);
    }

    @Test
    void aChangedHyperparameterIsRefused() throws IOException {
        storeFive(new Adam(RATE).persistAs(NAME, true));

        AbstractOptimizer other = new Adam(LearningRateSchedule.constant(RATE), 0.0f, 0.8f, 0.999f, 1e-8f)
                .persistAs(NAME, false);
        twoParameters().forEach(other::add);
        assertTrue(assertThrows(IllegalStateException.class, other::loadState).getMessage().contains("beta1"));
    }

    @Test
    void aChangedMomentumIsRefused() throws IOException {
        storeFive(new Sgd(RATE, 0.9f).persistAs(NAME, true));

        AbstractOptimizer other = new Sgd(RATE, 0.5f).persistAs(NAME, false);
        twoParameters().forEach(other::add);
        assertTrue(assertThrows(IllegalStateException.class, other::loadState).getMessage().contains("momentum"));
    }

    @Test
    void loadingWithoutANameIsRefused() {
        Adam adam = new Adam(RATE);
        twoParameters().forEach(adam::add);
        assertThrows(IllegalStateException.class, adam::loadState);
    }

    @Test
    void aMissingFileIsReportedWithItsPath() {
        AbstractOptimizer adam = new Adam(RATE).persistAs(NAME, false);
        twoParameters().forEach(adam::add);
        assertTrue(assertThrows(IllegalStateException.class, adam::loadState).getMessage().contains("o_" + NAME));
    }

    @Test
    void aFileThatIsNotOptimizerStateIsRefused() throws IOException {
        Files.createDirectories(Paths.get("./data/"));
        Files.write(PROMOTED, new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 });

        AbstractOptimizer adam = new Adam(RATE).persistAs(NAME, false);
        twoParameters().forEach(adam::add);
        assertThrows(IllegalStateException.class, adam::loadState);
    }

    @Test
    void nothingIsWrittenWithoutANameOrWithoutTheFlag() throws IOException {
        Adam unnamed = new Adam(RATE);
        twoParameters().forEach(unnamed::add);
        unnamed.storeState();
        assertFalse(Files.exists(WRITTEN), "an optimizer without a name must write nothing");

        AbstractOptimizer named = new Adam(RATE).persistAs(NAME, false);
        twoParameters().forEach(named::add);
        named.storeState();
        assertFalse(Files.exists(WRITTEN), "the store flag has to work the way the one of a layer does");
    }

    @Test
    void theHalfWrittenFileIsNotLeftBehind() throws IOException {
        storeFive(new Adam(RATE).persistAs(NAME, true));

        assertTrue(Files.exists(WRITTEN));
        assertFalse(Files.exists(PARTIAL), "the state is written under a temporary name and then moved");
    }

    @Test
    void theNetworkWritesTheOptimizerAlongWithTheLayers() throws IOException {
        Net net = new Net();
        net.add(new Hidden(4, 3, "optimizer_state_test_layer", false, false, 41L));
        net.optimizer(new Adam(RATE).persistAs(NAME, true));

        net.storeParameters();

        assertTrue(Files.exists(WRITTEN), "storeParameters() has to reach the optimizer too");
    }

    private static void storeFive(AbstractOptimizer optimizer) throws IOException {
        List<Parameter> params = twoParameters();
        params.forEach(optimizer::add);
        stepThrough(optimizer, params, 1, 5);
        optimizer.storeState();
        promote();
    }

    private static void promote() throws IOException {
        Files.createDirectories(Paths.get("./data/"));
        Files.copy(WRITTEN, PROMOTED, StandardCopyOption.REPLACE_EXISTING);
    }

    private static List<Parameter> twoParameters() {
        return List.of(new Parameter("w", Matrices.randomUniformF(4, 3, -1.0f, 1.0f, 61L), true),
                new Parameter("b", Matrices.randomUniformF(4, 1, -1.0f, 1.0f, 62L), false));
    }

    // what a layer does when it loads its weights from the checkpoint, without any of the layers
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
