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

import static math.ml.mlp.GradientCheck.assertInputGradient;
import static math.ml.mlp.GradientCheck.input;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import net.jamu.matrix.MatrixF;

class MeanPoolTest {

    private static final int FEATURES = 4;
    private static final int BLOCK = 3;
    private static final int BATCH = 2;

    @Test
    void forwardIsTheArithmeticMeanOfEachBlock() {
        MeanPool pool = new MeanPool(BLOCK);
        pool.setMode(NetworkMode.TRAIN);
        MatrixF x = input(FEATURES, BATCH * BLOCK, 11L);
        MatrixF y = pool.forward(x);

        assertEquals(FEATURES, y.numRows());
        assertEquals(BATCH, y.numColumns());
        for (int s = 0; s < BATCH; ++s) {
            for (int r = 0; r < FEATURES; ++r) {
                double sum = 0.0;
                for (int c = s * BLOCK; c < (s + 1) * BLOCK; ++c) {
                    sum += x.getUnsafe(r, c);
                }
                assertEquals(sum / BLOCK, y.getUnsafe(r, s), 1e-5, "block " + s + ", row " + r);
            }
        }
    }

    @Test
    void backwardMatchesNumericalInputGradient() {
        assertInputGradient(new MeanPool(BLOCK), input(FEATURES, BATCH * BLOCK, 13L), input(FEATURES, BATCH, 17L),
                2e-2);
    }

    @Test
    void backwardGivesEveryColumnOfABlockTheSameShare() {
        MeanPool pool = new MeanPool(BLOCK);
        pool.setMode(NetworkMode.TRAIN);
        pool.forward(input(FEATURES, BATCH * BLOCK, 19L));
        MatrixF g = input(FEATURES, BATCH, 23L);
        MatrixF dx = pool.backward(g);

        assertEquals(BATCH * BLOCK, dx.numColumns());
        for (int s = 0; s < BATCH; ++s) {
            for (int c = s * BLOCK; c < (s + 1) * BLOCK; ++c) {
                for (int r = 0; r < FEATURES; ++r) {
                    assertEquals(g.getUnsafe(r, s) / BLOCK, dx.getUnsafe(r, c), 1e-6f,
                            "column " + c + ", row " + r);
                }
            }
        }
    }

    // over the conv layout a block is one sample's spatial positions, which makes this
    // layer global average pooling without anything being added to it
    @Test
    void overTheConvLayoutItAveragesEachChannelOverTheImage() {
        Conv2D conv = new Conv2D(1, 2, 4, 4, 3, "mp_conv", Init.HE, 29L);
        MeanPool pool = new MeanPool(conv.outputHeight() * conv.outputWidth());
        conv.setMode(NetworkMode.TRAIN);
        pool.setMode(NetworkMode.TRAIN);

        MatrixF y = conv.forward(input(1, BATCH * 16, 31L));
        MatrixF pooled = pool.forward(y);

        assertEquals(2, pooled.numRows());
        assertEquals(BATCH, pooled.numColumns());
        int spatial = conv.outputHeight() * conv.outputWidth();
        for (int s = 0; s < BATCH; ++s) {
            for (int channel = 0; channel < 2; ++channel) {
                double sum = 0.0;
                for (int p = 0; p < spatial; ++p) {
                    sum += y.getUnsafe(channel, s * spatial + p);
                }
                assertEquals(sum / spatial, pooled.getUnsafe(channel, s), 1e-5, "channel " + channel);
            }
        }
    }

    @Test
    void aColumnCountThatIsNotAWholeNumberOfBlocksIsRejected() {
        MeanPool pool = new MeanPool(BLOCK);
        assertThrows(IllegalArgumentException.class, () -> pool.forward(input(FEATURES, BATCH * BLOCK + 1, 37L)));
    }

    @Test
    void aNonPositiveBlockSizeIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> new MeanPool(0));
    }

    @Test
    void backwardReturnsNullInInferenceMode() {
        MeanPool pool = new MeanPool(BLOCK);
        pool.setMode(NetworkMode.INFER);
        pool.forward(input(FEATURES, BATCH * BLOCK, 41L));
        assertNull(pool.backward(input(FEATURES, BATCH, 43L)));
    }
}
