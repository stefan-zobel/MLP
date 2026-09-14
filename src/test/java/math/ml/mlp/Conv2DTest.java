package math.ml.mlp;

import static math.ml.mlp.GradientCheck.H;
import static math.ml.mlp.GradientCheck.assertInputGradient;
import static math.ml.mlp.GradientCheck.dot;
import static math.ml.mlp.GradientCheck.grad;
import static math.ml.mlp.GradientCheck.input;
import static math.ml.mlp.GradientCheck.relativeError;
import static math.ml.mlp.GradientCheck.value;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

class Conv2DTest {

    private static final int IN_C = 2;
    private static final int OUT_C = 3;
    private static final int IN_H = 5;
    private static final int IN_W = 5;
    private static final int K = 3;
    private static final int BATCH = 2;

    // A convolution written out as its definition, with no matrix product anywhere. A
    // gradient check cannot replace this: central differences agree with a forward pass
    // that computes the wrong thing just as happily as with one that does not.
    private static MatrixF reference(MatrixF in, MatrixF kernels, MatrixF biases, int inC, int outC, int inH, int inW,
            int k, int stride, int pad) {
        int outH = (inH + 2 * pad - k) / stride + 1;
        int outW = (inW + 2 * pad - k) / stride + 1;
        int m = in.numColumns() / (inH * inW);
        MatrixF out = Matrices.createF(outC, m * outH * outW);
        for (int s = 0; s < m; ++s) {
            for (int oy = 0; oy < outH; ++oy) {
                for (int ox = 0; ox < outW; ++ox) {
                    for (int oc = 0; oc < outC; ++oc) {
                        double sum = biases.getUnsafe(oc, 0);
                        for (int c = 0; c < inC; ++c) {
                            for (int ky = 0; ky < k; ++ky) {
                                int iy = oy * stride - pad + ky;
                                if (iy < 0 || iy >= inH) {
                                    continue;
                                }
                                for (int kx = 0; kx < k; ++kx) {
                                    int ix = ox * stride - pad + kx;
                                    if (ix < 0 || ix >= inW) {
                                        continue;
                                    }
                                    sum += (double) kernels.getUnsafe(oc, (c * k + ky) * k + kx)
                                            * in.getUnsafe(c, s * inH * inW + iy * inW + ix);
                                }
                            }
                        }
                        out.setUnsafe(oc, s * outH * outW + oy * outW + ox, (float) sum);
                    }
                }
            }
        }
        return out;
    }

    private static Conv2D layer(int stride, int pad, long seed) throws Exception {
        Conv2D conv = new Conv2D(IN_C, OUT_C, IN_H, IN_W, K, stride, pad, "conv_test", Init.HE, seed);
        // a zero bias would let a bias indexing error through unnoticed
        MatrixF b = value(conv, "biases");
        for (int r = 0; r < OUT_C; ++r) {
            b.setUnsafe(r, 0, 0.25f * (r + 1));
        }
        return conv;
    }

    private static void assertMatchesReference(int stride, int pad, long seed) throws Exception {
        Conv2D conv = layer(stride, pad, seed);
        MatrixF x = input(IN_C, BATCH * IN_H * IN_W, seed + 1);
        MatrixF expected = reference(x, value(conv, "kernels"), value(conv, "biases"), IN_C, OUT_C, IN_H, IN_W, K,
                stride, pad);

        conv.setMode(NetworkMode.TRAIN);
        MatrixF actual = conv.forward(x);

        assertEquals(expected.numRows(), actual.numRows());
        assertEquals(expected.numColumns(), actual.numColumns());
        for (int c = 0; c < expected.numColumns(); ++c) {
            for (int r = 0; r < expected.numRows(); ++r) {
                assertEquals(expected.getUnsafe(r, c), actual.getUnsafe(r, c), 1e-4f,
                        "stride " + stride + ", pad " + pad + ", element [" + r + "," + c + "]");
            }
        }
    }

    @Test
    void forwardMatchesADirectConvolution() throws Exception {
        assertMatchesReference(1, 0, 11L);
    }

    @Test
    void forwardMatchesADirectConvolutionWithAStride() throws Exception {
        assertMatchesReference(2, 0, 13L);
    }

    @Test
    void forwardMatchesADirectConvolutionWithPadding() throws Exception {
        assertMatchesReference(1, 1, 17L);
    }

    @Test
    void forwardMatchesADirectConvolutionWithBothAtOnce() throws Exception {
        assertMatchesReference(2, 1, 19L);
    }

    @Test
    void theOutputGeometryFollowsTheKernelStrideAndPad() {
        assertEquals(3, new Conv2D(1, 1, 5, 5, 3, "g1", 1L).outputHeight());
        assertEquals(5, new Conv2D(1, 1, 5, 5, 3, 1, 1, "g2", Init.HE, 1L).outputHeight());
        assertEquals(2, new Conv2D(1, 1, 5, 5, 3, 2, 0, "g3", Init.HE, 1L).outputWidth());
        assertEquals(4, new Conv2D(1, 4, 5, 5, 3, "g4", 1L).outputChannels());
    }

    @Test
    void backwardMatchesNumericalInputGradient() {
        Conv2D conv = new Conv2D(IN_C, OUT_C, 4, 4, K, "conv_in", Init.HE, 23L);
        assertInputGradient(conv, input(IN_C, BATCH * 16, 29L), input(OUT_C, BATCH * 4, 31L), 2e-2);
    }

    @Test
    void backwardMatchesNumericalInputGradientWithPadding() {
        Conv2D conv = new Conv2D(IN_C, OUT_C, 4, 4, K, 1, 1, "conv_in_pad", Init.HE, 37L);
        assertInputGradient(conv, input(IN_C, BATCH * 16, 41L), input(OUT_C, BATCH * 16, 43L), 2e-2);
    }

    @Test
    void kernelAndBiasUpdatesMatchNumericalGradients() throws Exception {
        MatrixF x = input(IN_C, BATCH * IN_H * IN_W, 47L);
        MatrixF w = input(OUT_C, BATCH * 3 * 3, 53L);

        Conv2D conv = layer(1, 0, 59L);
        conv.setMode(NetworkMode.TRAIN);
        conv.forward(x);
        conv.backward(w.copy());

        MatrixF dKernels = grad(conv, "kernels");
        MatrixF dBiases = grad(conv, "biases");
        MatrixF kernels = value(conv, "kernels");
        MatrixF biases = value(conv, "biases");
        for (int r = 0; r < OUT_C; ++r) {
            // the buffers hold the mean over the batch; the numerical check sums
            double analyticBias = dBiases.getUnsafe(r, 0) * BATCH;
            assertTrue(relativeError(numericalGradient(conv, x, w, biases, r, 0), analyticBias) <= 2e-2,
                    "bias gradient mismatch in row " + r);
            for (int c = 0; c < kernels.numColumns(); ++c) {
                double analyticKernel = dKernels.getUnsafe(r, c) * BATCH;
                assertTrue(relativeError(numericalGradient(conv, x, w, kernels, r, c), analyticKernel) <= 2e-2,
                        "kernel gradient mismatch at [" + r + "," + c + "]");
            }
        }
    }

    private static double numericalGradient(Conv2D conv, MatrixF x, MatrixF w, MatrixF parameter, int row, int col) {
        float original = parameter.getUnsafe(row, col);
        parameter.setUnsafe(row, col, original + H);
        double plus = dot(w, conv.forward(x).copy());
        parameter.setUnsafe(row, col, original - H);
        double minus = dot(w, conv.forward(x).copy());
        parameter.setUnsafe(row, col, original);
        return (plus - minus) / (2.0 * H);
    }

    @Test
    void backwardReturnsNullInInferenceMode() {
        Conv2D conv = new Conv2D(IN_C, OUT_C, IN_H, IN_W, K, "conv_infer", 61L);
        conv.setMode(NetworkMode.INFER);
        conv.forward(input(IN_C, BATCH * IN_H * IN_W, 67L));
        assertTrue(conv.backward(input(OUT_C, BATCH * 9, 71L)) == null);
    }

    @Test
    void theHeDrawUsesTheConvolutionalFanIn() {
        MatrixF k = new Conv2D(IN_C, OUT_C, IN_H, IN_W, K, 1, 0, "conv_he", Init.HE, 73L).kernels.value();
        float bound = (float) Math.sqrt(6.0 / (IN_C * K * K));
        for (int c = 0; c < k.numColumns(); ++c) {
            for (int r = 0; r < k.numRows(); ++r) {
                assertTrue(Math.abs(k.getUnsafe(r, c)) <= bound, "kernel [" + r + "," + c + "] left its bound");
            }
        }
    }

    @Test
    void theSameSeedProducesTheSameKernels() {
        MatrixF a = new Conv2D(IN_C, OUT_C, IN_H, IN_W, K, "conv_a", 79L).kernels.value();
        MatrixF b = new Conv2D(IN_C, OUT_C, IN_H, IN_W, K, "conv_b", 79L).kernels.value();
        for (int c = 0; c < a.numColumns(); ++c) {
            for (int r = 0; r < a.numRows(); ++r) {
                assertEquals(a.getUnsafe(r, c), b.getUnsafe(r, c));
            }
        }
    }

    @Test
    void aWrongChannelCountIsRejected() {
        Conv2D conv = new Conv2D(IN_C, OUT_C, IN_H, IN_W, K, "conv_c", 83L);
        assertThrows(IllegalArgumentException.class, () -> conv.forward(input(IN_C + 1, IN_H * IN_W, 89L)));
    }

    @Test
    void aColumnCountThatIsNotAWholeNumberOfImagesIsRejected() {
        Conv2D conv = new Conv2D(IN_C, OUT_C, IN_H, IN_W, K, "conv_d", 97L);
        assertThrows(IllegalArgumentException.class, () -> conv.forward(input(IN_C, IN_H * IN_W + 1, 101L)));
    }

    @Test
    void aGeometryThatLeavesNothingIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> new Conv2D(1, 1, 3, 3, 5, "conv_e", 103L));
    }

    // a stride larger than the shortfall makes the truncating division report one output
    // position for a kernel that does not fit anywhere
    @Test
    void aKernelThatDoesNotFitIsRejectedWhateverTheStride() {
        assertThrows(IllegalArgumentException.class,
                () -> new Conv2D(1, 1, 3, 3, 5, 4, 0, "conv_e2", Init.HE, 104L));
    }

    @Test
    void aNonPositiveStrideIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> new Conv2D(1, 1, 5, 5, 3, 0, 0, "conv_f", Init.HE, 107L));
    }

    @Test
    void aNegativePadIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> new Conv2D(1, 1, 5, 5, 3, 1, -1, "conv_g", Init.HE, 109L));
    }

    // The conv layout puts every channel in a row and every sample-and-position pair in a
    // column, which is what makes a plain BatchNorm over it per-channel spatial batch norm.
    // Asserted rather than assumed, because the whole stack depends on it.
    @Test
    void aBatchNormOverTheConvLayoutNormalizesPerChannel() {
        Conv2D conv = new Conv2D(IN_C, OUT_C, IN_H, IN_W, K, "conv_bn", Init.HE, 113L);
        BatchNorm norm = new BatchNorm(OUT_C);
        conv.setMode(NetworkMode.TRAIN);
        norm.setMode(NetworkMode.TRAIN);

        MatrixF y = norm.forward(conv.forward(input(IN_C, 8 * IN_H * IN_W, 127L)));
        for (int r = 0; r < OUT_C; ++r) {
            double sum = 0.0;
            double sumSquares = 0.0;
            for (int c = 0; c < y.numColumns(); ++c) {
                sum += y.getUnsafe(r, c);
                sumSquares += (double) y.getUnsafe(r, c) * y.getUnsafe(r, c);
            }
            int n = y.numColumns();
            assertEquals(0.0, sum / n, 1e-3, "channel " + r + " is not centered over batch and space");
            assertEquals(1.0, sumSquares / n, 1e-2, "channel " + r + " is not scaled over batch and space");
        }
    }
}
