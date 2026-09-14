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

import static math.ml.mlp.GradientCheck.H;
import static math.ml.mlp.GradientCheck.assertInputGradient;
import static math.ml.mlp.GradientCheck.dot;
import static math.ml.mlp.GradientCheck.grad;
import static math.ml.mlp.GradientCheck.input;
import static math.ml.mlp.GradientCheck.relativeError;
import static math.ml.mlp.GradientCheck.value;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import net.jamu.matrix.MatrixF;

class PositionalEncodingTest {

    private static final int D_MODEL = 5;
    private static final int SEQ_LEN = 3;
    private static final int BATCH = 4;

    @Test
    void everyBlockGetsTheSameTableAtTheSamePosition() throws Exception {
        PositionalEncoding layer = new PositionalEncoding(D_MODEL, SEQ_LEN, "pe_add", 11L);
        layer.setMode(NetworkMode.TRAIN);
        MatrixF x = input(D_MODEL, BATCH * SEQ_LEN, 13L);
        MatrixF table = value(layer, "positions");
        MatrixF y = layer.forward(x);

        for (int s = 0; s < BATCH; ++s) {
            for (int t = 0; t < SEQ_LEN; ++t) {
                int col = s * SEQ_LEN + t;
                for (int r = 0; r < D_MODEL; ++r) {
                    assertEquals(x.getUnsafe(r, col) + table.getUnsafe(r, t), y.getUnsafe(r, col), 1e-6f,
                            "sample " + s + ", position " + t + ", row " + r);
                }
            }
        }
    }

    @Test
    void backwardPassesTheInputGradientThrough() {
        assertInputGradient(new PositionalEncoding(D_MODEL, SEQ_LEN, "pe_in", 17L),
                input(D_MODEL, BATCH * SEQ_LEN, 19L), input(D_MODEL, BATCH * SEQ_LEN, 23L), 2e-2);
    }

    @Test
    void theTableUpdateMatchesItsNumericalGradient() throws Exception {
        MatrixF x = input(D_MODEL, BATCH * SEQ_LEN, 29L);
        MatrixF w = input(D_MODEL, BATCH * SEQ_LEN, 31L);

        PositionalEncoding layer = new PositionalEncoding(D_MODEL, SEQ_LEN, "pe_par", 37L);
        layer.setMode(NetworkMode.TRAIN);
        layer.forward(x);
        layer.backward(w.copy());

        MatrixF table = value(layer, "positions");
        MatrixF gradient = grad(layer, "positions");
        for (int t = 0; t < SEQ_LEN; ++t) {
            for (int r = 0; r < D_MODEL; ++r) {
                // the buffer holds the mean over the samples; the numerical check sums
                double analytic = gradient.getUnsafe(r, t) * BATCH;
                float original = table.getUnsafe(r, t);
                table.setUnsafe(r, t, original + H);
                double plus = dot(w, layer.forward(x).copy());
                table.setUnsafe(r, t, original - H);
                double minus = dot(w, layer.forward(x).copy());
                table.setUnsafe(r, t, original);
                assertTrue(relativeError((plus - minus) / (2.0 * H), analytic) <= 2e-2,
                        "position " + t + ", row " + r);
            }
        }
    }

    @Test
    void aWrongFeatureCountIsRejected() {
        PositionalEncoding layer = new PositionalEncoding(D_MODEL, SEQ_LEN, "pe_feat", 41L);
        assertThrows(IllegalArgumentException.class, () -> layer.forward(input(D_MODEL + 1, SEQ_LEN, 43L)));
    }

    @Test
    void aColumnCountThatIsNotAWholeNumberOfSequencesIsRejected() {
        PositionalEncoding layer = new PositionalEncoding(D_MODEL, SEQ_LEN, "pe_cols", 47L);
        assertThrows(IllegalArgumentException.class, () -> layer.forward(input(D_MODEL, SEQ_LEN + 1, 53L)));
    }

    @Test
    void aNonPositiveDimensionIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> new PositionalEncoding(0, SEQ_LEN, "pe_zero", 59L));
        assertThrows(IllegalArgumentException.class, () -> new PositionalEncoding(D_MODEL, 0, "pe_zero2", 61L));
    }

    @Test
    void backwardReturnsNullInInferenceMode() {
        PositionalEncoding layer = new PositionalEncoding(D_MODEL, SEQ_LEN, "pe_infer", 67L);
        layer.setMode(NetworkMode.INFER);
        layer.forward(input(D_MODEL, BATCH * SEQ_LEN, 71L));
        assertNull(layer.backward(input(D_MODEL, BATCH * SEQ_LEN, 73L)));
    }

    @Test
    void theSameSeedProducesTheSameTable() throws Exception {
        MatrixF a = value(new PositionalEncoding(D_MODEL, SEQ_LEN, "pe_s1", 79L), "positions");
        MatrixF b = value(new PositionalEncoding(D_MODEL, SEQ_LEN, "pe_s2", 79L), "positions");
        for (int c = 0; c < a.numColumns(); ++c) {
            for (int r = 0; r < a.numRows(); ++r) {
                assertEquals(a.getUnsafe(r, c), b.getUnsafe(r, c));
            }
        }
    }
}
