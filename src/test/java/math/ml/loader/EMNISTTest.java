package math.ml.loader;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

public class EMNISTTest {

    private static final int W = 5;
    private static final int H = 3;

    // asymmetric in both axes and pairwise distinct, so a rotation or a mirroring cannot
    // pass for a transpose
    private static byte[] pattern() {
        byte[] image = new byte[W * H];
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                image[y * W + x] = (byte) (10 * y + x);
            }
        }
        return image;
    }

    @Test
    public void transposeSwapsRowsAndColumns() {
        byte[] out = EMNIST.transpose(pattern(), W, H);
        // the result is H wide and W tall
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                assertEquals(10 * y + x, out[x * H + y], "pixel " + x + ", " + y);
            }
        }
    }

    @Test
    public void transposeTwiceIsTheIdentity() {
        byte[] image = pattern();
        byte[] there = EMNIST.transpose(image, W, H);
        // width and height have traded places, so the way back reads them the other way round
        byte[] back = EMNIST.transpose(there, H, W);
        assertArrayEquals(image, back);
    }

    @Test
    public void transposeMovesACornerToTheOppositeCorner() {
        byte[] image = new byte[W * H];
        image[W - 1] = (byte) 255;
        byte[] out = EMNIST.transpose(image, W, H);
        assertEquals((byte) 255, out[(W - 1) * H]);
    }

    @Test
    public void rejectsALengthThatIsNotTheProduct() {
        byte[] image = new byte[W * H + 1];
        assertThrows(IllegalArgumentException.class, () -> EMNIST.transpose(image, W, H));
    }
}
