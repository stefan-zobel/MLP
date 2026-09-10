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
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import net.jamu.matrix.MatrixF;

class HiddenTest {

    private static final int IN = 12;
    private static final int OUT = 7;

    @Test
    void theSameSeedProducesTheSameWeights() {
        MatrixF a = new Hidden(IN, OUT, "a", 4242L).weights.value();
        MatrixF b = new Hidden(IN, OUT, "b", 4242L).weights.value();

        assertEquals(a.numRows(), b.numRows());
        assertEquals(a.numColumns(), b.numColumns());
        for (int c = 0; c < a.numColumns(); ++c) {
            for (int r = 0; r < a.numRows(); ++r) {
                assertEquals(a.getUnsafe(r, c), b.getUnsafe(r, c), 0.0f, "differs at " + r + "," + c);
            }
        }
    }

    @Test
    void aDifferentSeedProducesDifferentWeights() {
        // without this the reproducibility test above would also pass if the seed
        // were silently ignored
        assertTrue(differs(new Hidden(IN, OUT, "a", 4242L).weights.value(), new Hidden(IN, OUT, "b", 4243L).weights.value()));
    }

    @Test
    void theSeedlessConstructorStaysUnseeded() {
        assertTrue(differs(new Hidden(IN, OUT, "a").weights.value(), new Hidden(IN, OUT, "b").weights.value()));
    }

    @Test
    void theGlorotDrawStaysWithinItsBound() {
        assertWithinBound(new Hidden(IN, OUT, "a", 17L).weights.value(), (float) Math.sqrt(6.0 / (IN + OUT)));
    }

    @Test
    void theHeDrawStaysWithinItsBound() {
        assertWithinBound(new Hidden(IN, OUT, "a", Init.HE, 17L).weights.value(), (float) Math.sqrt(6.0 / IN));
    }

    @Test
    void heDiffersFromGlorotOnlyByTheBound() {
        // both map the same uniform stream onto [-bound, bound], so with one seed the
        // draws must be proportional -- this pins down that Init changes nothing else
        float ratio = (float) Math.sqrt((double) (IN + OUT) / IN);
        MatrixF glorot = new Hidden(IN, OUT, "a", Init.GLOROT, 23L).weights.value();
        MatrixF he = new Hidden(IN, OUT, "b", Init.HE, 23L).weights.value();

        for (int c = 0; c < glorot.numColumns(); ++c) {
            for (int r = 0; r < glorot.numRows(); ++r) {
                assertEquals(glorot.getUnsafe(r, c) * ratio, he.getUnsafe(r, c), 1e-6f,
                        "not proportional at " + r + "," + c);
            }
        }
        assertTrue(ratio > 1.0f, "He must draw wider than Glorot");
    }

    @Test
    void theSeedlessAndSeededConstructorsDefaultToGlorot() {
        // the default must not move: every call site that predates Init keeps its scheme
        float bound = (float) Math.sqrt(6.0 / (IN + OUT));
        assertWithinBound(new Hidden(IN, OUT, "a", 29L).weights.value(), bound);
        assertWithinBound(new Hidden(IN, OUT, "a", false, false, 29L).weights.value(), bound);
        assertWithinBound(new Hidden(IN, OUT, "a").weights.value(), bound);
    }

    private static void assertWithinBound(MatrixF w, float bound) {
        for (int c = 0; c < w.numColumns(); ++c) {
            for (int r = 0; r < w.numRows(); ++r) {
                assertTrue(Math.abs(w.getUnsafe(r, c)) <= bound, "out of bounds at " + r + "," + c);
            }
        }
    }

    private static boolean differs(MatrixF a, MatrixF b) {
        for (int c = 0; c < a.numColumns(); ++c) {
            for (int r = 0; r < a.numRows(); ++r) {
                if (a.getUnsafe(r, c) != b.getUnsafe(r, c)) {
                    return true;
                }
            }
        }
        return false;
    }
}
