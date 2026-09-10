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
        MatrixF a = new Hidden(IN, OUT, "a", 4242L).weights;
        MatrixF b = new Hidden(IN, OUT, "b", 4242L).weights;

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
        assertTrue(differs(new Hidden(IN, OUT, "a", 4242L).weights, new Hidden(IN, OUT, "b", 4243L).weights));
    }

    @Test
    void theSeedlessConstructorStaysUnseeded() {
        assertTrue(differs(new Hidden(IN, OUT, "a").weights, new Hidden(IN, OUT, "b").weights));
    }

    @Test
    void theDrawStaysWithinTheGlorotBound() {
        float bound = (float) Math.sqrt(6.0 / (IN + OUT));
        MatrixF w = new Hidden(IN, OUT, "a", 17L).weights;
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
