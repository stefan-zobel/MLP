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
import static math.ml.mlp.GradientCheck.dot;
import static math.ml.mlp.GradientCheck.input;
import static math.ml.mlp.GradientCheck.relativeError;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

/**
 * Every layer of the encoder passes its own gradient check, which says nothing about the stack:
 * a wrong sequence length, a residual branch on the wrong side or a block out of order all keep
 * the shapes intact and fail silently. These are the checks that see them.
 *
 * <p>Nothing here compares bit for bit. The stack contains matrix products, and their result
 * depends on the alignment of the destination, so two runs of the same arithmetic agree to about
 * a unit in the last place and no further.
 */
class TransformerEncoderTest {

    private static final int IMAGE_SIZE = 28;
    private static final int TILE = 14;
    private static final int SEQ_LEN = 4;
    private static final int D_MODEL = 8;
    private static final int HEADS = 4;
    private static final int BLOCKS = 2;
    private static final int BATCH = 3;

    @Test
    void theSequenceLengthIsTheConvolutionsOutputSpatialSize() {
        TransformerEncoder encoder = encoder(1L);
        assertEquals(SEQ_LEN, encoder.sequenceLength());
        assertEquals(D_MODEL, encoder.features());
    }

    @Test
    void aBatchOfImagesBecomesABatchOfPooledVectors() {
        TransformerEncoder encoder = encoder(11L);
        encoder.setMode(NetworkMode.TRAIN);
        MatrixF pooled = encoder.forward(images(BATCH, 13L));

        assertEquals(D_MODEL, pooled.numRows());
        assertEquals(BATCH, pooled.numColumns());
    }

    // The single most likely wiring error is a sequence length that does not match the
    // convolution: MeanPool would then average across sample boundaries, PositionalEncoding
    // would repeat at the wrong period and Attention would mix samples. None of that changes
    // a shape, so nothing else in the suite would notice. The perturbation is large on purpose,
    // so that a leak is orders of magnitude above the noise the matrix products leave.
    @Test
    void oneImageCannotInfluenceThePooledVectorOfAnother() {
        TransformerEncoder encoder = encoder(17L);
        encoder.setMode(NetworkMode.TRAIN);

        MatrixF x = images(BATCH, 19L);
        MatrixF before = encoder.forward(x).copy();

        MatrixF changed = x.copy();
        for (int r = 0; r < changed.numRows(); ++r) {
            changed.setUnsafe(r, 2, changed.getUnsafe(r, 2) + 5.0f);
        }
        MatrixF after = encoder.forward(changed);

        for (int c = 0; c < 2; ++c) {
            for (int r = 0; r < D_MODEL; ++r) {
                assertEquals(before.getUnsafe(r, c), after.getUnsafe(r, c), 1e-4f,
                        "sample " + c + " moved when sample 2 changed, at row " + r);
            }
        }
    }

    // Not assertInputGradient: its relative error floors at 1e-3, which is right for a single
    // layer but sits inside the noise of a float loss accumulated through fifteen of them.
    // Measured over the step sizes, the error grows as the step shrinks, which is cancellation
    // and not truncation, and it is confined to the entries near that floor. So the entries that
    // carry a gradient are checked, and the count of them is asserted as well, because a filter
    // that quietly kept nothing would pass.
    @Test
    void theEntriesThatCarryAGradientMatchCentralDifferences() {
        TransformerEncoder encoder = encoder(23L);
        MatrixF x = images(2, 29L);
        MatrixF w = input(D_MODEL, 2, 31L);

        encoder.setMode(NetworkMode.TRAIN);
        encoder.forward(x.copy());
        MatrixF analytic = encoder.backward(w.copy());

        int checked = 0;
        double worst = 0.0;
        int worstRow = -1;
        int worstCol = -1;
        for (int c = 0; c < x.numColumns(); ++c) {
            for (int r = 0; r < x.numRows(); ++r) {
                double a = analytic.getUnsafe(r, c);
                if (Math.abs(a) < 5e-3) {
                    continue;
                }
                ++checked;
                double numeric = (perturbedLoss(encoder, x, w, r, c, H) - perturbedLoss(encoder, x, w, r, c, -H))
                        / (2.0 * H);
                double error = relativeError(numeric, a);
                if (error > worst) {
                    worst = error;
                    worstRow = r;
                    worstCol = c;
                }
            }
        }

        assertTrue(checked >= 200, "only " + checked + " entries carried a gradient worth checking");
        double maxError = worst;
        int row = worstRow;
        int col = worstCol;
        int entries = checked;
        assertTrue(maxError <= 3e-2, () -> "max relative error " + maxError + " at [" + row + "," + col
                + "] over " + entries + " entries");
    }

    // The counterpart of the equivariance AttentionTest pins on the bare layer: attention is
    // permutation-equivariant and the pooled mean is symmetric, so without a positional encoding
    // this stack would see a bag of tiles and swapping two of them would change nothing. A table
    // that degenerates to a single position is silent in every other way.
    @Test
    void swappingTwoTilesOfAnImageChangesItsOutput() {
        TransformerEncoder encoder = encoder(59L);
        encoder.setMode(NetworkMode.TRAIN);

        MatrixF x = images(1, 61L);
        MatrixF swapped = x.copy();
        for (int y = 0; y < TILE; ++y) {
            for (int col = 0; col < TILE; ++col) {
                int a = y * IMAGE_SIZE + col;
                int b = (y + TILE) * IMAGE_SIZE + col + TILE;
                float keep = swapped.getUnsafe(a, 0);
                swapped.setUnsafe(a, 0, swapped.getUnsafe(b, 0));
                swapped.setUnsafe(b, 0, keep);
            }
        }

        MatrixF before = encoder.forward(x).copy();
        MatrixF after = encoder.forward(swapped);
        double worst = 0.0;
        for (int r = 0; r < D_MODEL; ++r) {
            worst = Math.max(worst, Math.abs(before.getUnsafe(r, 0) - after.getUnsafe(r, 0)));
        }
        assertTrue(worst > 1e-3, "the output moved by only " + worst + ", so the tiles carry no position");
    }

    @Test
    void theSameSeedBuildsTheSameEncoder() {
        TransformerEncoder one = encoder(37L);
        TransformerEncoder two = encoder(37L);
        one.setMode(NetworkMode.TRAIN);
        two.setMode(NetworkMode.TRAIN);

        MatrixF x = images(BATCH, 41L);
        MatrixF a = one.forward(x).copy();
        MatrixF b = two.forward(x);
        for (int c = 0; c < a.numColumns(); ++c) {
            for (int r = 0; r < a.numRows(); ++r) {
                assertEquals(a.getUnsafe(r, c), b.getUnsafe(r, c), 1e-5f, "[" + r + "," + c + "]");
            }
        }
    }

    @Test
    void backwardReturnsNullInInferenceMode() {
        TransformerEncoder encoder = encoder(43L);
        encoder.setMode(NetworkMode.INFER);
        encoder.forward(images(BATCH, 47L));
        assertNull(encoder.backward(input(D_MODEL, BATCH, 53L)));
    }

    // -----------------------------------------------------------------------
    // What the geometry may be, now that it is not hard-coded
    // -----------------------------------------------------------------------

    @Test
    void aNonSquareImageWorks() {
        TransformerEncoder encoder = TransformerEncoder.builder().image(12, 20).tile(4).dModel(D_MODEL)
                .heads(HEADS).blocks(1).activation(Sigmoid::new).names("te_rect").seed(67L).build();
        assertEquals(15, encoder.sequenceLength());

        encoder.setMode(NetworkMode.TRAIN);
        MatrixF pooled = encoder.forward(Matrices.randomUniformF(12 * 20, BATCH, 0.0f, 1.0f, 71L));
        assertEquals(D_MODEL, pooled.numRows());
        assertEquals(BATCH, pooled.numColumns());
    }

    @Test
    void moreThanOneChannelWorks() {
        TransformerEncoder encoder = TransformerEncoder.builder().image(8, 8).channels(3).tile(4).dModel(D_MODEL)
                .heads(HEADS).blocks(1).activation(Sigmoid::new).names("te_rgb").seed(73L).build();
        assertEquals(4, encoder.sequenceLength());

        encoder.setMode(NetworkMode.TRAIN);
        MatrixF pooled = encoder.forward(Matrices.randomUniformF(3 * 8 * 8, BATCH, 0.0f, 1.0f, 79L));
        assertEquals(D_MODEL, pooled.numRows());
        assertEquals(BATCH, pooled.numColumns());
    }

    // Conv2D accepts this and quietly drops the remainder, which is correct for a convolution
    // and a trap for a patch projection: three rows of the image would never be seen.
    @Test
    void aTileThatDoesNotDivideTheImageIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> base().tile(5).names("te_bad").seed(83L).build());
    }

    @Test
    void headsThatDoNotDivideTheModelWidthAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> base().heads(3).names("te_heads").seed(89L).build());
    }

    @Test
    void aNonPositiveDimensionIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> base().channels(0).names("te_ch").seed(97L).build());
        assertThrows(IllegalArgumentException.class, () -> base().mlpWidth(0).names("te_mlp").seed(101L).build());
    }

    @Test
    void aSettingWithoutADefaultMustBeGiven() {
        assertThrows(IllegalStateException.class,
                () -> TransformerEncoder.builder().tile(TILE).dModel(D_MODEL).heads(HEADS).blocks(BLOCKS)
                        .names("te_noimage").seed(103L).build());
        assertThrows(IllegalStateException.class, () -> base().seed(107L).build());
        assertThrows(IllegalStateException.class, () -> base().names("te_noseed").build());
    }

    // 4 * dModel is the usual width and the one the run uses, so it is the default; the parameter
    // count is what distinguishes it from any other choice.
    @Test
    void theFeedForwardWidthDefaultsToFourTimesTheModelWidth() {
        int implicit = parameterCount(base().names("te_dflt").seed(109L).build());
        int explicit = parameterCount(base().mlpWidth(4 * D_MODEL).names("te_expl").seed(109L).build());
        int narrow = parameterCount(base().mlpWidth(2 * D_MODEL).names("te_narrow").seed(109L).build());

        assertEquals(explicit, implicit);
        assertTrue(narrow < implicit, "a narrower feed-forward must have fewer weights");
    }

    private static TransformerEncoder.Builder base() {
        return TransformerEncoder.builder().image(IMAGE_SIZE, IMAGE_SIZE).tile(TILE).dModel(D_MODEL).heads(HEADS)
                .blocks(BLOCKS).activation(Sigmoid::new);
    }

    /** Sigmoid throughout: central differences are only valid on a smooth function. */
    private static TransformerEncoder encoder(long seed) {
        return base().names("te_" + seed).seed(seed).build();
    }

    private static int parameterCount(TransformerEncoder encoder) {
        int total = 0;
        for (Parameter p : encoder.parameters()) {
            total += p.value().numRows() * p.value().numColumns();
        }
        return total;
    }

    private static double perturbedLoss(Layer layer, MatrixF x, MatrixF w, int row, int col, float delta) {
        MatrixF perturbed = x.copy();
        perturbed.setUnsafe(row, col, x.getUnsafe(row, col) + delta);
        layer.setMode(NetworkMode.TRAIN);
        return dot(w, layer.forward(perturbed));
    }

    private static MatrixF images(int batch, long seed) {
        return Matrices.randomUniformF(IMAGE_SIZE * IMAGE_SIZE, batch, 0.0f, 1.0f, seed);
    }

    // Written out rather than read off, because this is the one check that sees a builder which
    // mixes two of its settings up: heads and blocks are both small integers, and the layers
    // come out self-consistent either way. Heads alone would not show -- the projections of h
    // heads of width dModel/h hold the same weights whatever h is -- but the block count does.
    @Test
    void theWeightCountIsWhatTheGeometrySays() {
        int dHead = D_MODEL / HEADS;
        int mlp = 4 * D_MODEL;
        int patch = D_MODEL * TILE * TILE + D_MODEL;
        int positions = D_MODEL * SEQ_LEN;
        int norm = 2 * D_MODEL;
        int attention = HEADS * (3 * dHead * D_MODEL + D_MODEL * dHead);
        int feedForward = (D_MODEL * mlp + mlp) + (mlp * D_MODEL + D_MODEL);
        int expected = patch + positions + BLOCKS * (norm + attention + norm + feedForward) + norm;

        assertEquals(expected, parameterCount(encoder(113L)));
    }
}
