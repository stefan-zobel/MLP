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
package math.ml.loader;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

public class AugmentedSetTest {

    private static final int W = 28;
    private static final int H = 28;
    private static final int PIXELS = W * H;
    private static final int COUNT = 9;
    private static final int LABEL_ROWS = 10;

    private static byte[] sourceData() {
        byte[] data = new byte[COUNT * PIXELS];
        for (int i = 0; i < COUNT; ++i) {
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    // pairwise distinct and asymmetric in both axes
                    data[i * PIXELS + y * W + x] = (byte) ((3 * x + 7 * y + 29 * i) % 256);
                }
            }
        }
        return data;
    }

    private static MatrixF oneHotLabels() {
        MatrixF labels = Matrices.createF(LABEL_ROWS, COUNT);
        for (int i = 0; i < COUNT; ++i) {
            labels.set(i, i, 1.0f);
        }
        return labels;
    }

    private static AugmentedSet set() {
        return new AugmentedSet(sourceData(), COUNT, W, H, oneHotLabels());
    }

    private static int labelOf(AugmentedSet set, int column) {
        float[] labels = set.labels().getArrayUnsafe();
        for (int r = 0; r < LABEL_ROWS; ++r) {
            if (labels[column * LABEL_ROWS + r] == 1.0f) {
                return r;
            }
        }
        return -1;
    }

    // the source index whose pixels the column reproduces exactly, or -1 if it was distorted
    private static int undistortedSource(AugmentedSet set, byte[] data, int column) {
        float[] images = set.images().getArrayUnsafe();
        for (int i = 0; i < COUNT; ++i) {
            boolean equal = true;
            for (int p = 0; p < PIXELS && equal; ++p) {
                equal = images[column * PIXELS + p] == (data[i * PIXELS + p] & 0xFF) / 255.0f;
            }
            if (equal) {
                return i;
            }
        }
        return -1;
    }

    @Test
    public void everyPassUsesEveryImageExactlyOnce() {
        AugmentedSet set = set();
        SplittableRandom rnd = new SplittableRandom(7L);
        for (int pass = 0; pass < 20; ++pass) {
            set.regenerate(rnd);
            int[] labels = new int[COUNT];
            for (int j = 0; j < COUNT; ++j) {
                labels[j] = labelOf(set, j);
            }
            Arrays.sort(labels);
            for (int i = 0; i < COUNT; ++i) {
                assertEquals(i, labels[i], "pass " + pass + " is not a permutation");
            }
        }
    }

    @Test
    public void undistortedColumnsCarryTheirOwnLabel() {
        byte[] data = sourceData();
        AugmentedSet set = set();
        SplittableRandom rnd = new SplittableRandom(11L);
        int checked = 0;
        for (int pass = 0; pass < 100; ++pass) {
            set.regenerate(rnd);
            for (int j = 0; j < COUNT; ++j) {
                int source = undistortedSource(set, data, j);
                if (source >= 0) {
                    assertEquals(source, labelOf(set, j), "pass " + pass + ", column " + j);
                    ++checked;
                }
            }
        }
        // about one image in seven is passed through undistorted
        assertTrue(checked > 50, "only " + checked + " undistorted columns in 900");
    }

    @Test
    public void aPassIsReproducible() {
        AugmentedSet a = set();
        AugmentedSet b = set();
        a.regenerate(new SplittableRandom(3L));
        b.regenerate(new SplittableRandom(3L));
        assertArrayEquals(a.images().getArrayUnsafe(), b.images().getArrayUnsafe());
        assertArrayEquals(a.labels().getArrayUnsafe(), b.labels().getArrayUnsafe());
    }

    @Test
    public void successivePassesDiffer() {
        AugmentedSet set = set();
        SplittableRandom rnd = new SplittableRandom(5L);
        set.regenerate(rnd);
        float[] first = set.images().getArrayUnsafe().clone();
        set.regenerate(rnd);
        assertFalse(Arrays.equals(first, set.images().getArrayUnsafe()));
    }

    @Test
    public void valuesStayInTheUnitInterval() {
        AugmentedSet set = set();
        set.regenerate(new SplittableRandom(13L));
        for (float v : set.images().getArrayUnsafe()) {
            assertTrue(v >= 0.0f && v <= 1.0f, "value " + v);
        }
    }

    @Test
    public void rejectsALabelCountMismatch() {
        byte[] data = sourceData();
        MatrixF labels = Matrices.createF(LABEL_ROWS, COUNT - 1);
        assertThrows(IllegalArgumentException.class, () -> new AugmentedSet(data, COUNT, W, H, labels));
    }

    @Test
    public void rejectsAMismatchedSourceLength() {
        MatrixF labels = oneHotLabels();
        assertThrows(IllegalArgumentException.class, () -> new AugmentedSet(new byte[10], COUNT, W, H, labels));
    }
    @Test
    public void identityWeightsOnlyLeaveEveryImageAlone() {
        byte[] data = sourceData();
        AugmentedSet set = new AugmentedSet(data, COUNT, W, H, oneHotLabels(), 1, 0, 0);
        SplittableRandom rnd = new SplittableRandom(17L);
        for (int pass = 0; pass < 10; ++pass) {
            set.regenerate(rnd);
            for (int j = 0; j < COUNT; ++j) {
                int source = undistortedSource(set, data, j);
                assertTrue(source >= 0, "pass " + pass + ", column " + j + " was distorted");
                assertEquals(source, labelOf(set, j), "pass " + pass + ", column " + j);
            }
        }
    }

    @Test
    public void rejectsWeightsThatAreAllZero() {
        byte[] data = sourceData();
        MatrixF labels = oneHotLabels();
        assertThrows(IllegalArgumentException.class, () -> new AugmentedSet(data, COUNT, W, H, labels, 0, 0, 0));
    }

    @Test
    public void rejectsANegativeWeight() {
        byte[] data = sourceData();
        MatrixF labels = oneHotLabels();
        assertThrows(IllegalArgumentException.class, () -> new AugmentedSet(data, COUNT, W, H, labels, 1, -1, 3));
    }
}
