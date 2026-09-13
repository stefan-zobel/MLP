package math.ml.mlp;

import static math.ml.mlp.GradientCheck.assertInputGradient;
import static math.ml.mlp.GradientCheck.input;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

class UnflattenTest {

    private static final int C = 2;
    private static final int H = 2;
    private static final int W = 3;
    private static final int SPATIAL = H * W;
    private static final int M = 2;

    // a value that names its own sample, channel and position
    private static MatrixF tagged() {
        MatrixF in = Matrices.createF(C * SPATIAL, M);
        for (int s = 0; s < M; ++s) {
            for (int c = 0; c < C; ++c) {
                for (int p = 0; p < SPATIAL; ++p) {
                    in.setUnsafe(c * SPATIAL + p, s, 100 * s + 10 * c + p);
                }
            }
        }
        return in;
    }

    @Test
    void oneColumnBecomesOneImagePerChannel() {
        Unflatten unflatten = new Unflatten(C, H, W);
        unflatten.setMode(NetworkMode.TRAIN);
        MatrixF out = unflatten.forward(tagged());

        assertEquals(C, out.numRows());
        assertEquals(M * SPATIAL, out.numColumns());
        assertEquals(C * SPATIAL, unflatten.features());
        for (int s = 0; s < M; ++s) {
            for (int c = 0; c < C; ++c) {
                for (int p = 0; p < SPATIAL; ++p) {
                    assertEquals(100 * s + 10 * c + p, out.getUnsafe(c, s * SPATIAL + p),
                            "sample " + s + ", channel " + c + ", position " + p);
                }
            }
        }
    }

    @Test
    void flattenAfterUnflattenIsTheIdentity() {
        Unflatten unflatten = new Unflatten(C, H, W);
        Flatten flatten = new Flatten(C, H, W);
        unflatten.setMode(NetworkMode.TRAIN);
        flatten.setMode(NetworkMode.TRAIN);

        MatrixF in = input(C * SPATIAL, M, 401L);
        MatrixF out = flatten.forward(unflatten.forward(in));

        assertEquals(in.numRows(), out.numRows());
        assertEquals(in.numColumns(), out.numColumns());
        for (int c = 0; c < in.numColumns(); ++c) {
            for (int r = 0; r < in.numRows(); ++r) {
                assertEquals(in.getUnsafe(r, c), out.getUnsafe(r, c), "element [" + r + "," + c + "]");
            }
        }
    }

    @Test
    void unflattenAfterFlattenIsTheIdentity() {
        Flatten flatten = new Flatten(C, H, W);
        Unflatten unflatten = new Unflatten(C, H, W);
        flatten.setMode(NetworkMode.TRAIN);
        unflatten.setMode(NetworkMode.TRAIN);

        MatrixF in = input(C, M * SPATIAL, 409L);
        MatrixF out = unflatten.forward(flatten.forward(in));

        for (int c = 0; c < in.numColumns(); ++c) {
            for (int r = 0; r < in.numRows(); ++r) {
                assertEquals(in.getUnsafe(r, c), out.getUnsafe(r, c), "element [" + r + "," + c + "]");
            }
        }
    }

    @Test
    void backwardPutsEveryElementBackWhereItCameFrom() {
        Unflatten unflatten = new Unflatten(C, H, W);
        unflatten.setMode(NetworkMode.TRAIN);
        MatrixF in = tagged();
        MatrixF back = unflatten.backward(unflatten.forward(in));

        assertEquals(C * SPATIAL, back.numRows());
        assertEquals(M, back.numColumns());
        for (int c = 0; c < M; ++c) {
            for (int r = 0; r < C * SPATIAL; ++r) {
                assertEquals(in.getUnsafe(r, c), back.getUnsafe(r, c), "element [" + r + "," + c + "]");
            }
        }
    }

    // the layout a loader hands over and the layout a single-channel convolution wants
    // hold their elements in the same order, which is what keeps the network entry cheap
    @Test
    void aSingleChannelKeepsTheElementOrder() {
        Unflatten unflatten = new Unflatten(1, H, W);
        unflatten.setMode(NetworkMode.TRAIN);
        MatrixF in = input(SPATIAL, M, 419L);
        MatrixF out = unflatten.forward(in);

        assertEquals(1, out.numRows());
        assertEquals(M * SPATIAL, out.numColumns());
        float[] source = in.getArrayUnsafe();
        float[] target = out.getArrayUnsafe();
        for (int i = 0; i < source.length; ++i) {
            assertEquals(source[i], target[i], "element " + i);
        }
    }

    @Test
    void backwardMatchesNumericalInputGradient() {
        assertInputGradient(new Unflatten(C, H, W), input(C * SPATIAL, M, 421L), input(C, M * SPATIAL, 431L), 2e-2);
    }

    @Test
    void theOutputFeedsAConvolutionUnchanged() {
        Unflatten unflatten = new Unflatten(1, 5, 5);
        Conv2D conv = new Conv2D(1, 3, 5, 5, 3, "unflat_conv", Init.HE, 433L);
        unflatten.setMode(NetworkMode.TRAIN);
        conv.setMode(NetworkMode.TRAIN);

        MatrixF out = conv.forward(unflatten.forward(input(25, M, 439L)));
        assertEquals(3, out.numRows());
        assertEquals(M * 3 * 3, out.numColumns());
    }

    @Test
    void backwardReturnsNullInInferenceMode() {
        Unflatten unflatten = new Unflatten(C, H, W);
        unflatten.setMode(NetworkMode.INFER);
        unflatten.forward(tagged());
        assertTrue(unflatten.backward(Matrices.createF(C, M * SPATIAL)) == null);
    }

    @Test
    void aWrongFeatureCountIsRejected() {
        Unflatten unflatten = new Unflatten(C, H, W);
        assertThrows(IllegalArgumentException.class, () -> unflatten.forward(input(C * SPATIAL + 1, M, 443L)));
    }

    @Test
    void aNonPositiveGeometryIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> new Unflatten(0, H, W));
        assertThrows(IllegalArgumentException.class, () -> new Unflatten(C, 0, W));
        assertThrows(IllegalArgumentException.class, () -> new Unflatten(C, H, 0));
    }
}
