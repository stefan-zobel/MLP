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

public class FrozenAugmentedSetTest {

    private static final int W = 28;
    private static final int H = 28;
    private static final int PIXELS = W * H;
    private static final int COUNT = 9;
    private static final int LABEL_ROWS = 10;
    private static final int PASSES = 3;

    private static byte[] sourceData() {
        byte[] data = new byte[COUNT * PIXELS];
        for (int i = 0; i < COUNT; ++i) {
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
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

    private static AugmentedSet live() {
        return new AugmentedSet(sourceData(), COUNT, W, H, oneHotLabels());
    }

    private static AugmentedSet undistortedLive() {
        return new AugmentedSet(sourceData(), COUNT, W, H, oneHotLabels(), 1, 0, 0);
    }

    private static FrozenAugmentedSet frozen(AugmentedSet source, long seed) {
        return FrozenAugmentedSet.of(source, PASSES, new SplittableRandom(seed));
    }

    private static int labelOf(PassSource set, int column) {
        float[] labels = set.labels().getArrayUnsafe();
        for (int r = 0; r < LABEL_ROWS; ++r) {
            if (labels[column * LABEL_ROWS + r] == 1.0f) {
                return r;
            }
        }
        return -1;
    }

    // the pixels each label carries in the pass just served, which is the pass content with
    // the column order divided out
    private static float[][] byLabel(PassSource set) {
        float[][] pixels = new float[LABEL_ROWS][];
        float[] images = set.images().getArrayUnsafe();
        for (int j = 0; j < COUNT; ++j) {
            pixels[labelOf(set, j)] = Arrays.copyOfRange(images, j * PIXELS, (j + 1) * PIXELS);
        }
        return pixels;
    }

    private static int[] labelOrder(PassSource set) {
        int[] order = new int[COUNT];
        for (int j = 0; j < COUNT; ++j) {
            order[j] = labelOf(set, j);
        }
        return order;
    }

    @Test
    public void everyPassUsesEveryImageExactlyOnce() {
        FrozenAugmentedSet set = frozen(live(), 7L);
        SplittableRandom rnd = new SplittableRandom(21L);
        for (int pass = 0; pass < 3 * PASSES; ++pass) {
            set.regenerate(rnd);
            int[] labels = labelOrder(set);
            Arrays.sort(labels);
            for (int i = 0; i < COUNT; ++i) {
                assertEquals(i, labels[i], "pass " + pass + " is not a permutation");
            }
        }
    }

    @Test
    public void passesComeRoundWithTheirPeriod() {
        FrozenAugmentedSet set = frozen(live(), 7L);
        SplittableRandom rnd = new SplittableRandom(23L);
        float[][][] seen = new float[2 * PASSES][][];
        for (int pass = 0; pass < 2 * PASSES; ++pass) {
            set.regenerate(rnd);
            seen[pass] = byLabel(set);
        }
        for (int p = 0; p < PASSES; ++p) {
            for (int label = 0; label < COUNT; ++label) {
                assertArrayEquals(seen[p][label], seen[p + PASSES][label], "pass " + p + ", label " + label);
            }
        }
    }

    @Test
    public void differentPassesHoldDifferentImages() {
        FrozenAugmentedSet set = frozen(live(), 7L);
        SplittableRandom rnd = new SplittableRandom(29L);
        set.regenerate(rnd);
        float[][] first = byLabel(set);
        set.regenerate(rnd);
        float[][] second = byLabel(set);
        boolean differs = false;
        for (int label = 0; label < COUNT && !differs; ++label) {
            differs = !Arrays.equals(first[label], second[label]);
        }
        assertTrue(differs, "two distinct frozen passes hold the same images");
    }

    @Test
    public void theColumnOrderIsDrawnAnew() {
        FrozenAugmentedSet set = frozen(live(), 7L);
        SplittableRandom rnd = new SplittableRandom(31L);
        set.regenerate(rnd);
        int[] first = labelOrder(set);
        for (int p = 0; p < PASSES; ++p) {
            set.regenerate(rnd);
        }
        // back at the same pass, so only the order can have moved
        assertFalse(Arrays.equals(first, labelOrder(set)));
    }

    @Test
    public void imageAndLabelStayTogether() {
        byte[] data = sourceData();
        FrozenAugmentedSet set = frozen(undistortedLive(), 37L);
        SplittableRandom rnd = new SplittableRandom(41L);
        for (int pass = 0; pass < 2 * PASSES; ++pass) {
            set.regenerate(rnd);
            float[] images = set.images().getArrayUnsafe();
            for (int j = 0; j < COUNT; ++j) {
                int label = labelOf(set, j);
                for (int p = 0; p < PIXELS; ++p) {
                    assertEquals((data[label * PIXELS + p] & 0xFF) / 255.0f, images[j * PIXELS + p],
                            "pass " + pass + ", column " + j + ", pixel " + p);
                }
            }
        }
    }

    @Test
    public void rejectsANonPositivePassCount() {
        AugmentedSet source = live();
        SplittableRandom rnd = new SplittableRandom(43L);
        assertThrows(IllegalArgumentException.class, () -> FrozenAugmentedSet.of(source, 0, rnd));
    }
}
