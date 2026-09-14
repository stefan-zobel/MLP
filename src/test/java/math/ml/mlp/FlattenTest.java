package math.ml.mlp;

import static math.ml.mlp.GradientCheck.assertInputGradient;
import static math.ml.mlp.GradientCheck.input;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

class FlattenTest {

    private static final int C = 2;
    private static final int H = 2;
    private static final int W = 3;
    private static final int SPATIAL = H * W;
    private static final int M = 2;

    // a value that names its own sample, channel and position
    private static MatrixF tagged() {
        MatrixF in = Matrices.createF(C, M * SPATIAL);
        for (int s = 0; s < M; ++s) {
            for (int p = 0; p < SPATIAL; ++p) {
                for (int c = 0; c < C; ++c) {
                    in.setUnsafe(c, s * SPATIAL + p, 100 * s + 10 * c + p);
                }
            }
        }
        return in;
    }

    @Test
    void oneSampleBecomesOneColumnOfChannelBlocks() {
        Flatten flatten = new Flatten(C, H, W);
        flatten.setMode(NetworkMode.TRAIN);
        MatrixF out = flatten.forward(tagged());

        assertEquals(C * SPATIAL, out.numRows());
        assertEquals(M, out.numColumns());
        assertEquals(C * SPATIAL, flatten.features());
        for (int s = 0; s < M; ++s) {
            for (int c = 0; c < C; ++c) {
                for (int p = 0; p < SPATIAL; ++p) {
                    assertEquals(100 * s + 10 * c + p, out.getUnsafe(c * SPATIAL + p, s),
                            "sample " + s + ", channel " + c + ", position " + p);
                }
            }
        }
    }

    @Test
    void backwardPutsEveryElementBackWhereItCameFrom() {
        Flatten flatten = new Flatten(C, H, W);
        flatten.setMode(NetworkMode.TRAIN);
        MatrixF in = tagged();
        MatrixF back = flatten.backward(flatten.forward(in));

        assertEquals(C, back.numRows());
        assertEquals(M * SPATIAL, back.numColumns());
        for (int col = 0; col < M * SPATIAL; ++col) {
            for (int c = 0; c < C; ++c) {
                assertEquals(in.getUnsafe(c, col), back.getUnsafe(c, col), "element [" + c + "," + col + "]");
            }
        }
    }

    @Test
    void aSingleChannelIsACopy() {
        Flatten flatten = new Flatten(1, H, W);
        flatten.setMode(NetworkMode.TRAIN);
        MatrixF in = input(1, M * SPATIAL, 307L);
        MatrixF out = flatten.forward(in);

        assertEquals(SPATIAL, out.numRows());
        assertEquals(M, out.numColumns());
        float[] source = in.getArrayUnsafe();
        float[] target = out.getArrayUnsafe();
        for (int i = 0; i < source.length; ++i) {
            assertEquals(source[i], target[i], "element " + i);
        }
    }

    @Test
    void backwardMatchesNumericalInputGradient() {
        assertInputGradient(new Flatten(C, H, W), input(C, M * SPATIAL, 311L), input(C * SPATIAL, M, 313L), 2e-2);
    }

    @Test
    void theOutputFeedsAHiddenLayerUnchanged() {
        Flatten flatten = new Flatten(C, H, W);
        Hidden hidden = new Hidden(C * SPATIAL, 4, "flat_hidden", 317L);
        flatten.setMode(NetworkMode.TRAIN);
        hidden.setMode(NetworkMode.TRAIN);

        MatrixF out = hidden.forward(flatten.forward(tagged()));
        assertEquals(4, out.numRows());
        assertEquals(M, out.numColumns());
    }

    @Test
    void backwardReturnsNullInInferenceMode() {
        Flatten flatten = new Flatten(C, H, W);
        flatten.setMode(NetworkMode.INFER);
        flatten.forward(tagged());
        assertTrue(flatten.backward(Matrices.createF(C * SPATIAL, M)) == null);
    }

    @Test
    void aWrongChannelCountIsRejected() {
        Flatten flatten = new Flatten(C, H, W);
        assertThrows(IllegalArgumentException.class, () -> flatten.forward(input(C + 1, SPATIAL, 331L)));
    }

    @Test
    void aColumnCountThatIsNotAWholeNumberOfImagesIsRejected() {
        Flatten flatten = new Flatten(C, H, W);
        assertThrows(IllegalArgumentException.class, () -> flatten.forward(input(C, SPATIAL + 1, 337L)));
    }

    @Test
    void aNonPositiveGeometryIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> new Flatten(0, H, W));
        assertThrows(IllegalArgumentException.class, () -> new Flatten(C, 0, W));
        assertThrows(IllegalArgumentException.class, () -> new Flatten(C, H, 0));
    }
}
