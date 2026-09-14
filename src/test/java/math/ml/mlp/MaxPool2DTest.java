package math.ml.mlp;

import static math.ml.mlp.GradientCheck.assertInputGradient;
import static math.ml.mlp.GradientCheck.input;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

class MaxPool2DTest {

    private static final int H = 4;
    private static final int W = 4;

    // one channel, one sample, pixel (y, x) carries the value y * W + x
    private static MatrixF ramp() {
        MatrixF in = Matrices.createF(1, H * W);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                in.setUnsafe(0, y * W + x, y * W + x);
            }
        }
        return in;
    }

    @Test
    void eachWindowContributesItsLargestElement() {
        MaxPool2D pool = new MaxPool2D(1, H, W, 2);
        pool.setMode(NetworkMode.TRAIN);
        MatrixF out = pool.forward(ramp());

        assertEquals(2, pool.outputHeight());
        assertEquals(2, pool.outputWidth());
        assertEquals(4, out.numColumns());
        assertEquals(5.0f, out.getUnsafe(0, 0));
        assertEquals(7.0f, out.getUnsafe(0, 1));
        assertEquals(13.0f, out.getUnsafe(0, 2));
        assertEquals(15.0f, out.getUnsafe(0, 3));
    }

    @Test
    void theGradientReachesOnlyTheWinningPositions() {
        MaxPool2D pool = new MaxPool2D(1, H, W, 2);
        pool.setMode(NetworkMode.TRAIN);
        pool.forward(ramp());

        MatrixF grads = Matrices.createF(1, 4);
        for (int c = 0; c < 4; ++c) {
            grads.setUnsafe(0, c, c + 1.0f);
        }
        MatrixF back = pool.backward(grads);

        assertEquals(H * W, back.numColumns());
        for (int p = 0; p < H * W; ++p) {
            float expected = switch (p) {
            case 5 -> 1.0f;
            case 7 -> 2.0f;
            case 13 -> 3.0f;
            case 15 -> 4.0f;
            default -> 0.0f;
            };
            assertEquals(expected, back.getUnsafe(0, p), "position " + p);
        }
    }

    @Test
    void aTieGoesToTheLowestIndex() {
        MatrixF flat = Matrices.createF(1, H * W);
        // every element of the first window is the same, so only the rule decides
        flat.setUnsafe(0, 0, 3.0f);
        flat.setUnsafe(0, 1, 3.0f);
        flat.setUnsafe(0, 4, 3.0f);
        flat.setUnsafe(0, 5, 3.0f);

        MaxPool2D pool = new MaxPool2D(1, H, W, 2);
        pool.setMode(NetworkMode.TRAIN);
        pool.forward(flat);

        MatrixF grads = Matrices.createF(1, 4);
        grads.setUnsafe(0, 0, 1.0f);
        MatrixF back = pool.backward(grads);

        assertEquals(1.0f, back.getUnsafe(0, 0));
        assertEquals(0.0f, back.getUnsafe(0, 1));
        assertEquals(0.0f, back.getUnsafe(0, 4));
        assertEquals(0.0f, back.getUnsafe(0, 5));
    }

    @Test
    void anAllNegativeWindowKeepsItsLargestElement() {
        MatrixF negative = Matrices.createF(1, H * W);
        for (int p = 0; p < H * W; ++p) {
            negative.setUnsafe(0, p, -(p + 1.0f));
        }
        MaxPool2D pool = new MaxPool2D(1, H, W, 2);
        pool.setMode(NetworkMode.TRAIN);
        MatrixF out = pool.forward(negative);
        // -1 is the largest of -1, -2, -5, -6, which a zero-initialized maximum would miss
        assertEquals(-1.0f, out.getUnsafe(0, 0));
    }

    @Test
    void channelsAndSamplesStaySeparate() {
        int channels = 2;
        int m = 2;
        MatrixF in = Matrices.createF(channels, m * H * W);
        for (int s = 0; s < m; ++s) {
            for (int p = 0; p < H * W; ++p) {
                for (int c = 0; c < channels; ++c) {
                    // a value that identifies its sample, channel and position uniquely
                    in.setUnsafe(c, s * H * W + p, 100 * s + 10 * c + p);
                }
            }
        }
        MaxPool2D pool = new MaxPool2D(channels, H, W, 2);
        pool.setMode(NetworkMode.TRAIN);
        MatrixF out = pool.forward(in);

        assertEquals(m * 4, out.numColumns());
        for (int s = 0; s < m; ++s) {
            for (int c = 0; c < channels; ++c) {
                assertEquals(100 * s + 10 * c + 5, out.getUnsafe(c, s * 4), "sample " + s + ", channel " + c);
                assertEquals(100 * s + 10 * c + 15, out.getUnsafe(c, s * 4 + 3), "sample " + s + ", channel " + c);
            }
        }
    }

    @Test
    void anOverlappingWindowIsAllowed() {
        MaxPool2D pool = new MaxPool2D(1, H, W, 2, 1);
        assertEquals(3, pool.outputHeight());
        assertEquals(3, pool.outputWidth());
        pool.setMode(NetworkMode.TRAIN);
        MatrixF out = pool.forward(ramp());
        assertEquals(5.0f, out.getUnsafe(0, 0));
        assertEquals(15.0f, out.getUnsafe(0, 8));
    }

    @Test
    void backwardMatchesNumericalInputGradient() {
        assertInputGradient(new MaxPool2D(2, H, W, 2), input(2, 2 * H * W, 211L), input(2, 2 * 4, 223L), 2e-2);
    }

    @Test
    void backwardReturnsNullInInferenceMode() {
        MaxPool2D pool = new MaxPool2D(1, H, W, 2);
        pool.setMode(NetworkMode.INFER);
        pool.forward(ramp());
        assertTrue(pool.backward(Matrices.createF(1, 4)) == null);
    }

    @Test
    void aWrongChannelCountIsRejected() {
        MaxPool2D pool = new MaxPool2D(2, H, W, 2);
        assertThrows(IllegalArgumentException.class, () -> pool.forward(input(3, H * W, 227L)));
    }

    @Test
    void aColumnCountThatIsNotAWholeNumberOfImagesIsRejected() {
        MaxPool2D pool = new MaxPool2D(1, H, W, 2);
        assertThrows(IllegalArgumentException.class, () -> pool.forward(input(1, H * W + 1, 229L)));
    }

    // a stride larger than the shortfall makes the truncating division report one output
    // position for a window that does not fit anywhere
    @Test
    void aWindowLargerThanTheInputIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> new MaxPool2D(1, 3, 3, 4));
        assertThrows(IllegalArgumentException.class, () -> new MaxPool2D(1, 3, 3, 4, 4));
    }
}
