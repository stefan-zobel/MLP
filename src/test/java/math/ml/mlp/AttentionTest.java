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
import static math.ml.mlp.GradientCheck.field;
import static math.ml.mlp.GradientCheck.relativeError;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

class AttentionTest {

    private static final int D_MODEL = 6;
    private static final int SEQ_LEN = 3;
    private static final int HEADS = 2;
    private static final int BATCH = 2;

    // Scores grow with the input, and a saturated softmax is a bad place to compare float
    // against double or to take a central difference, so these stay away from +/-2.
    private static MatrixF small(int rows, int cols, long seed) {
        return Matrices.randomUniformF(rows, cols, -0.8f, 0.8f, seed);
    }

    // Attention written out as its definition: every projection, score, softmax and context
    // as an explicit loop in double, with no matrix product anywhere. Central differences
    // cannot replace this, being just as content with a forward pass that attends wrongly.
    private static MatrixF reference(MatrixF in, Parameter[] wq, Parameter[] wk, Parameter[] wv, Parameter[] wo,
            int dModel, int seqLen, int heads) {
        int dHead = dModel / heads;
        int m = in.numColumns() / seqLen;
        double scale = 1.0 / Math.sqrt(dHead);
        MatrixF out = Matrices.createF(dModel, in.numColumns());
        for (int s = 0; s < m; ++s) {
            for (int h = 0; h < heads; ++h) {
                double[][] q = new double[seqLen][dHead];
                double[][] k = new double[seqLen][dHead];
                double[][] v = new double[seqLen][dHead];
                for (int t = 0; t < seqLen; ++t) {
                    int col = s * seqLen + t;
                    for (int d = 0; d < dHead; ++d) {
                        double sq = 0.0;
                        double sk = 0.0;
                        double sv = 0.0;
                        for (int e = 0; e < dModel; ++e) {
                            double x = in.getUnsafe(e, col);
                            sq += wq[h].value().getUnsafe(d, e) * x;
                            sk += wk[h].value().getUnsafe(d, e) * x;
                            sv += wv[h].value().getUnsafe(d, e) * x;
                        }
                        q[t][d] = sq;
                        k[t][d] = sk;
                        v[t][d] = sv;
                    }
                }
                for (int j = 0; j < seqLen; ++j) {
                    double[] p = new double[seqLen];
                    double max = Double.NEGATIVE_INFINITY;
                    for (int i = 0; i < seqLen; ++i) {
                        double product = 0.0;
                        for (int d = 0; d < dHead; ++d) {
                            product += q[j][d] * k[i][d];
                        }
                        p[i] = product * scale;
                        max = Math.max(max, p[i]);
                    }
                    double sum = 0.0;
                    for (int i = 0; i < seqLen; ++i) {
                        p[i] = Math.exp(p[i] - max);
                        sum += p[i];
                    }
                    double[] context = new double[dHead];
                    for (int d = 0; d < dHead; ++d) {
                        double acc = 0.0;
                        for (int i = 0; i < seqLen; ++i) {
                            acc += v[i][d] * (p[i] / sum);
                        }
                        context[d] = acc;
                    }
                    int col = s * seqLen + j;
                    for (int r = 0; r < dModel; ++r) {
                        double acc = 0.0;
                        for (int d = 0; d < dHead; ++d) {
                            acc += wo[h].value().getUnsafe(r, d) * context[d];
                        }
                        out.setUnsafe(r, col, out.getUnsafe(r, col) + (float) acc);
                    }
                }
            }
        }
        return out;
    }

    private static void assertMatchesReference(int dModel, int seqLen, int heads, long seed) throws Exception {
        Attention layer = new Attention(dModel, seqLen, heads, "att_ref", seed);
        MatrixF x = small(dModel, BATCH * seqLen, seed + 1);
        MatrixF expected = reference(x, field(layer, "queries"), field(layer, "keys"), field(layer, "values"),
                field(layer, "outputs"), dModel, seqLen, heads);

        layer.setMode(NetworkMode.TRAIN);
        MatrixF actual = layer.forward(x);

        assertEquals(expected.numRows(), actual.numRows());
        assertEquals(expected.numColumns(), actual.numColumns());
        for (int c = 0; c < expected.numColumns(); ++c) {
            for (int r = 0; r < expected.numRows(); ++r) {
                assertEquals(expected.getUnsafe(r, c), actual.getUnsafe(r, c), 2e-4f,
                        heads + " heads, element [" + r + "," + c + "]");
            }
        }
    }

    @Test
    void forwardMatchesTheDefinition() throws Exception {
        assertMatchesReference(D_MODEL, SEQ_LEN, HEADS, 11L);
    }

    @Test
    void forwardMatchesTheDefinitionWithOneHead() throws Exception {
        assertMatchesReference(D_MODEL, SEQ_LEN, 1, 13L);
    }

    @Test
    void forwardMatchesTheDefinitionWithAHeadPerFeature() throws Exception {
        assertMatchesReference(4, 5, 4, 17L);
    }

    @Test
    void forwardMatchesTheDefinitionAtSequenceLengthOne() throws Exception {
        assertMatchesReference(D_MODEL, 1, HEADS, 19L);
    }

    @Test
    void backwardMatchesNumericalInputGradient() {
        Attention layer = new Attention(D_MODEL, SEQ_LEN, HEADS, "att_in", 23L);
        assertInputGradient(layer, small(D_MODEL, BATCH * SEQ_LEN, 29L), small(D_MODEL, BATCH * SEQ_LEN, 31L), 3e-2);
    }

    @Test
    void backwardMatchesNumericalInputGradientWithOneHead() {
        Attention layer = new Attention(D_MODEL, SEQ_LEN, 1, "att_in1", 37L);
        assertInputGradient(layer, small(D_MODEL, BATCH * SEQ_LEN, 41L), small(D_MODEL, BATCH * SEQ_LEN, 43L), 3e-2);
    }

    @Test
    void everyProjectionUpdateMatchesItsNumericalGradient() throws Exception {
        MatrixF x = small(D_MODEL, BATCH * SEQ_LEN, 47L);
        MatrixF w = small(D_MODEL, BATCH * SEQ_LEN, 53L);

        Attention layer = new Attention(D_MODEL, SEQ_LEN, HEADS, "att_par", 59L);
        layer.setMode(NetworkMode.TRAIN);
        layer.forward(x);
        layer.backward(w.copy());

        for (String kind : new String[] { "queries", "keys", "values", "outputs" }) {
            Parameter[] group = field(layer, kind);
            for (int h = 0; h < group.length; ++h) {
                MatrixF value = group[h].value();
                MatrixF gradient = group[h].grad();
                for (int c = 0; c < value.numColumns(); ++c) {
                    for (int r = 0; r < value.numRows(); ++r) {
                        // the buffers hold the mean over the samples; the numerical check sums
                        double analytic = gradient.getUnsafe(r, c) * BATCH;
                        double numeric = numericalGradient(layer, x, w, value, r, c);
                        assertTrue(relativeError(numeric, analytic) <= 3e-2,
                                kind + " head " + h + " mismatch at [" + r + "," + c + "]");
                    }
                }
            }
        }
    }

    private static double numericalGradient(Attention layer, MatrixF x, MatrixF w, MatrixF parameter, int row,
            int col) {
        float original = parameter.getUnsafe(row, col);
        parameter.setUnsafe(row, col, original + H);
        double plus = dot(w, layer.forward(x).copy());
        parameter.setUnsafe(row, col, original - H);
        double minus = dot(w, layer.forward(x).copy());
        parameter.setUnsafe(row, col, original);
        return (plus - minus) / (2.0 * H);
    }

    @Test
    void everyQueryDistributesExactlyOneUnitOfAttention() throws Exception {
        Attention layer = new Attention(D_MODEL, SEQ_LEN, HEADS, "att_p", 61L);
        layer.setMode(NetworkMode.TRAIN);
        layer.forward(small(D_MODEL, BATCH * SEQ_LEN, 67L));

        float[][] probabilities = field(layer, "probabilities");
        for (int h = 0; h < HEADS; ++h) {
            for (int s = 0; s < BATCH; ++s) {
                for (int j = 0; j < SEQ_LEN; ++j) {
                    double sum = 0.0;
                    int off = (s * SEQ_LEN + j) * SEQ_LEN;
                    for (int i = 0; i < SEQ_LEN; ++i) {
                        float p = probabilities[h][off + i];
                        assertTrue(p > 0.0f && p <= 1.0f, "attention weight out of range: " + p);
                        sum += p;
                    }
                    assertEquals(1.0, sum, 1e-5, "head " + h + ", sample " + s + ", query " + j);
                }
            }
        }
    }

    // The sequence axis is folded into the column axis, so a block boundary is the one place
    // this layout can go wrong without any shape mismatch to show for it.
    @Test
    void oneSampleCannotAttendToAnother() {
        Attention layer = new Attention(D_MODEL, SEQ_LEN, HEADS, "att_block", 71L);
        layer.setMode(NetworkMode.TRAIN);
        MatrixF x = small(D_MODEL, BATCH * SEQ_LEN, 73L);
        MatrixF before = layer.forward(x).copy();

        MatrixF changed = x.copy();
        for (int c = SEQ_LEN; c < BATCH * SEQ_LEN; ++c) {
            for (int r = 0; r < D_MODEL; ++r) {
                changed.setUnsafe(r, c, changed.getUnsafe(r, c) + 1.5f);
            }
        }
        MatrixF after = layer.forward(changed);

        for (int c = 0; c < SEQ_LEN; ++c) {
            for (int r = 0; r < D_MODEL; ++r) {
                assertEquals(before.getUnsafe(r, c), after.getUnsafe(r, c), 1e-6f,
                        "the second sample moved element [" + r + "," + c + "] of the first");
            }
        }
    }

    // Without a positional encoding a sequence is a set, so permuting the tokens of a sample
    // must permute its output columns and change nothing else.
    @Test
    void attentionIsEquivariantUnderAPermutationOfTheTokens() {
        Attention layer = new Attention(D_MODEL, SEQ_LEN, HEADS, "att_perm", 79L);
        layer.setMode(NetworkMode.TRAIN);
        MatrixF x = small(D_MODEL, SEQ_LEN, 83L);
        MatrixF plain = layer.forward(x).copy();

        int[] permutation = { 2, 0, 1 };
        MatrixF shuffled = Matrices.createF(D_MODEL, SEQ_LEN);
        for (int c = 0; c < SEQ_LEN; ++c) {
            for (int r = 0; r < D_MODEL; ++r) {
                shuffled.setUnsafe(r, c, x.getUnsafe(r, permutation[c]));
            }
        }
        MatrixF permuted = layer.forward(shuffled);

        for (int c = 0; c < SEQ_LEN; ++c) {
            for (int r = 0; r < D_MODEL; ++r) {
                assertEquals(plain.getUnsafe(r, permutation[c]), permuted.getUnsafe(r, c), 1e-5f,
                        "element [" + r + "," + c + "] did not follow the permutation");
            }
        }
    }

    @Test
    void backwardReturnsNullInInferenceMode() {
        Attention layer = new Attention(D_MODEL, SEQ_LEN, HEADS, "att_infer", 89L);
        layer.setMode(NetworkMode.INFER);
        layer.forward(small(D_MODEL, BATCH * SEQ_LEN, 97L));
        assertNull(layer.backward(small(D_MODEL, BATCH * SEQ_LEN, 101L)));
    }

    @Test
    void everyHeadContributesFourProjections() {
        assertEquals(4 * HEADS, new Attention(D_MODEL, SEQ_LEN, HEADS, "att_count", 103L).parameters().size());
    }

    @Test
    void theSameSeedProducesTheSameProjections() throws Exception {
        Parameter[] a = field(new Attention(D_MODEL, SEQ_LEN, HEADS, "att_s1", 107L), "queries");
        Parameter[] b = field(new Attention(D_MODEL, SEQ_LEN, HEADS, "att_s2", 107L), "queries");
        for (int h = 0; h < HEADS; ++h) {
            for (int c = 0; c < a[h].value().numColumns(); ++c) {
                for (int r = 0; r < a[h].value().numRows(); ++r) {
                    assertEquals(a[h].value().getUnsafe(r, c), b[h].value().getUnsafe(r, c));
                }
            }
        }
    }

    // the heads of one layer must not all start from the same draw
    @Test
    void theHeadsDoNotShareAnInitialization() throws Exception {
        Parameter[] q = field(new Attention(D_MODEL, SEQ_LEN, HEADS, "att_heads", 109L), "queries");
        boolean different = false;
        for (int c = 0; c < q[0].value().numColumns() && !different; ++c) {
            for (int r = 0; r < q[0].value().numRows(); ++r) {
                if (q[0].value().getUnsafe(r, c) != q[1].value().getUnsafe(r, c)) {
                    different = true;
                    break;
                }
            }
        }
        assertTrue(different, "both heads drew the same query projection");
    }

    @Test
    void aHeadCountThatDoesNotDivideTheWidthIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> new Attention(6, 3, 4, "att_bad", 113L));
    }

    @Test
    void aNonPositiveDimensionIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> new Attention(6, 0, 2, "att_zero", 127L));
        assertThrows(IllegalArgumentException.class, () -> new Attention(6, 3, 0, "att_nohead", 131L));
    }

    @Test
    void aWrongFeatureCountIsRejected() {
        Attention layer = new Attention(D_MODEL, SEQ_LEN, HEADS, "att_feat", 137L);
        assertThrows(IllegalArgumentException.class, () -> layer.forward(small(D_MODEL + 1, SEQ_LEN, 139L)));
    }

    @Test
    void aColumnCountThatIsNotAWholeNumberOfSequencesIsRejected() {
        Attention layer = new Attention(D_MODEL, SEQ_LEN, HEADS, "att_cols", 149L);
        assertThrows(IllegalArgumentException.class, () -> layer.forward(small(D_MODEL, SEQ_LEN + 1, 151L)));
    }
}
