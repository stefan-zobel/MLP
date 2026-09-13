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

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

/**
 * A 2D convolution over a {@code channels x (batch * height * width)} layout, where every
 * pair of sample and spatial position is one column.
 *
 * <p>The patches are gathered into a matrix so that the convolution itself is the same
 * {@code sgemm} a {@link Hidden} performs. The spatial dimensions cannot be recovered from
 * that layout and are therefore declared rather than inferred.
 */
public class Conv2D extends AbstractLayer {

    /**
     * Curated parameters. Read-only from code, so that no training run can
     * overwrite a set that was promoted here by hand.
     */
    private static final String LOAD_DIR = "./data/";

    /**
     * Where training runs write. Promote a checkpoint to {@link #LOAD_DIR}
     * manually once it has proven itself.
     */
    private static final String STORE_DIR = "./checkpoints/";

    /** The kernels, out x (in * kernel * kernel), one row per output channel. */
    protected final Parameter kernels;
    /** The bias column, out x 1, one entry per output channel. */
    protected final Parameter biases;
    /** Identifies the parameter files of this layer. */
    protected final String name;
    /** Whether {@link #storeParameters()} writes anything. */
    protected final boolean storeKernelsAndBiases;

    private final int inChannels;
    private final int outChannels;
    private final int inH;
    private final int inW;
    private final int kernel;
    private final int stride;
    private final int pad;
    private final int outH;
    private final int outW;

    /** The gathered patches, kept for the backward pass; this is the cache, not the input. */
    private MatrixF patches;
    private MatrixF output;
    private MatrixF patchGrads;
    private MatrixF inputGrads;
    /** The batch size the buffers above were sized for, or -1 before the first pass. */
    private int batch = -1;

    /**
     * Creates a layer with stride 1, no padding and Glorot initialization.
     *
     * @param inChannels  channels of the input
     * @param outChannels channels of the output
     * @param inH         height of the input
     * @param inW         width of the input
     * @param kernel      edge length of the square kernel
     * @param name        identifies the parameter files {@code w_<name>} and {@code b_<name>}
     * @param seed        seed for the kernel draw
     */
    public Conv2D(int inChannels, int outChannels, int inH, int inW, int kernel, String name, long seed) {
        this(inChannels, outChannels, inH, inW, kernel, 1, 0, name, false, false, Init.GLOROT, seed);
    }

    /**
     * Creates a layer with stride 1 and no padding; use {@link Init#HE} when a ReLU or
     * GELU follows.
     *
     * @param inChannels  channels of the input
     * @param outChannels channels of the output
     * @param inH         height of the input
     * @param inW         width of the input
     * @param kernel      edge length of the square kernel
     * @param name        identifies the parameter files {@code w_<name>} and {@code b_<name>}
     * @param init        the kernel initialization scheme
     * @param seed        seed for the kernel draw
     */
    public Conv2D(int inChannels, int outChannels, int inH, int inW, int kernel, String name, Init init, long seed) {
        this(inChannels, outChannels, inH, inW, kernel, 1, 0, name, false, false, init, seed);
    }

    /**
     * Creates a layer that neither loads nor stores its parameters.
     *
     * @param inChannels  channels of the input
     * @param outChannels channels of the output
     * @param inH         height of the input
     * @param inW         width of the input
     * @param kernel      edge length of the square kernel
     * @param stride      step between neighboring patches
     * @param pad         zeros added on every side of the input
     * @param name        identifies the parameter files {@code w_<name>} and {@code b_<name>}
     * @param init        the kernel initialization scheme
     * @param seed        seed for the kernel draw
     */
    public Conv2D(int inChannels, int outChannels, int inH, int inW, int kernel, int stride, int pad, String name,
            Init init, long seed) {
        this(inChannels, outChannels, inH, inW, kernel, stride, pad, name, false, false, init, seed);
    }

    /**
     * The full form; use {@link Init#HE} when a ReLU or GELU follows.
     *
     * @param inChannels             channels of the input
     * @param outChannels            channels of the output
     * @param inH                    height of the input
     * @param inW                    width of the input
     * @param kernel                 edge length of the square kernel
     * @param stride                 step between neighboring patches
     * @param pad                    zeros added on every side of the input
     * @param name                   identifies the parameter files {@code w_<name>} and {@code b_<name>}
     * @param loadKernelsAndBiases   read the parameters from {@code ./data/} at construction
     * @param storeKernelsAndBiases  let {@link #storeParameters()} write to {@code ./checkpoints/}
     * @param init                   the kernel initialization scheme
     * @param seed                   seed for the kernel draw, unused when the parameters are loaded
     */
    public Conv2D(int inChannels, int outChannels, int inH, int inW, int kernel, int stride, int pad, String name,
            boolean loadKernelsAndBiases, boolean storeKernelsAndBiases, Init init, long seed) {
        if (inChannels <= 0 || outChannels <= 0) {
            throw new IllegalArgumentException("channels must be positive, got " + inChannels + " and " + outChannels);
        }
        if (kernel <= 0 || stride <= 0 || pad < 0) {
            throw new IllegalArgumentException(
                    "kernel " + kernel + ", stride " + stride + " and pad " + pad + " are not a usable geometry");
        }
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.inH = inH;
        this.inW = inW;
        this.kernel = kernel;
        this.stride = stride;
        this.pad = pad;
        int spanH = inH + 2 * pad - kernel;
        int spanW = inW + 2 * pad - kernel;
        // the span has to be tested before the division: integer division truncates
        // towards zero, so a kernel that does not fit at all still divides to a
        // positive count once the stride exceeds the shortfall
        if (spanH < 0 || spanW < 0) {
            throw new IllegalArgumentException("a " + kernel + "x" + kernel + " kernel with stride " + stride
                    + " and pad " + pad + " does not fit into a " + inH + "x" + inW + " input");
        }
        this.outH = spanH / stride + 1;
        this.outW = spanW / stride + 1;
        this.name = name;
        this.storeKernelsAndBiases = storeKernelsAndBiases;
        int patchRows = inChannels * kernel * kernel;
        MatrixF k;
        MatrixF b;
        if (loadKernelsAndBiases) {
            k = load(LOAD_DIR + "w_" + name);
            b = load(LOAD_DIR + "b_" + name);
        } else {
            // the canonical convolutional fan-in and fan-out: one kernel sees patchRows
            // inputs and contributes to outChannels * kernel * kernel outputs
            float bound = init.bound(patchRows, outChannels * kernel * kernel);
            k = Matrices.randomUniformF(outChannels, patchRows, -bound, bound, seed);
            b = Matrices.createF(outChannels, 1);
        }
        // weight decay applies to the kernels but not to the bias, the standard rule
        kernels = new Parameter("kernels", k, true);
        biases = new Parameter("biases", b, false);
    }

    /** Height of this layer's output. */
    public int outputHeight() {
        return outH;
    }

    /** Width of this layer's output. */
    public int outputWidth() {
        return outW;
    }

    /** Channels of this layer's output. */
    public int outputChannels() {
        return outChannels;
    }

    /** The kernels and the bias column. */
    @Override
    public List<Parameter> parameters() {
        return List.of(kernels, biases);
    }

    @Override
    public MatrixF forward(MatrixF in) {
        int m = batchOf(in);
        ensureBuffers(m);
        im2col(in.getArrayUnsafe(), patches.getArrayUnsafe(), m);
        // (out x patchRows) * (patchRows x m*outH*outW) = (out x m*outH*outW)
        kernels.value().mult(patches, output);
        return output.addBroadcastedVectorInplace(biases.value());
    }

    // outputGrads : out x (m * outH * outW)
    @Override
    public MatrixF backward(MatrixF outputGrads) {
        if (mode == NetworkMode.INFER) {
            return null;
        }
        int m = outputGrads.numColumns() / (outH * outW);
        // (patchRows x out) * (out x m*outH*outW) = (patchRows x m*outH*outW)
        kernels.value().transAmult(outputGrads, patchGrads);
        col2im(patchGrads.getArrayUnsafe(), inputGrads.getArrayUnsafe(), m);
        // One kernel is shared by every position, so its gradient is the sum over all of
        // them; dividing by m rather than by the column count is what makes that a mean
        // over the batch, which is the convention the other layers keep.
        outputGrads.transBmult(patches, kernels.grad()).scaleInplace(1.0f / m);
        // colsAverage divides by m * outH * outW, so the spatial factor goes back in
        biases.grad().setInplace(Matrices.colsAverage(outputGrads).scaleInplace((float) (outH * outW)));
        return inputGrads;
    }

    private int batchOf(MatrixF in) {
        if (in.numRows() != inChannels) {
            throw new IllegalArgumentException("expected " + inChannels + " channels, got " + in.numRows());
        }
        int spatial = inH * inW;
        if (in.numColumns() % spatial != 0) {
            throw new IllegalArgumentException(
                    in.numColumns() + " columns is not a whole number of " + inH + "x" + inW + " images");
        }
        return in.numColumns() / spatial;
    }

    private void ensureBuffers(int m) {
        if (batch == m) {
            return;
        }
        int patchRows = inChannels * kernel * kernel;
        patches = Matrices.createF(patchRows, m * outH * outW);
        output = Matrices.createF(outChannels, m * outH * outW);
        patchGrads = Matrices.createF(patchRows, m * outH * outW);
        inputGrads = Matrices.createF(inChannels, m * inH * inW);
        batch = m;
    }

    /**
     * Gathers every patch into its own column. Every element of the destination is written,
     * so a padded position needs no separate pass to clear it.
     */
    private void im2col(float[] src, float[] dst, int m) {
        int patchRows = inChannels * kernel * kernel;
        int inSpatial = inH * inW;
        int outSpatial = outH * outW;
        for (int s = 0; s < m; ++s) {
            int sample = s * inSpatial * inChannels;
            for (int oy = 0; oy < outH; ++oy) {
                for (int ox = 0; ox < outW; ++ox) {
                    int column = (s * outSpatial + oy * outW + ox) * patchRows;
                    int iy0 = oy * stride - pad;
                    int ix0 = ox * stride - pad;
                    for (int c = 0; c < inChannels; ++c) {
                        for (int ky = 0; ky < kernel; ++ky) {
                            int iy = iy0 + ky;
                            boolean rowInside = iy >= 0 && iy < inH;
                            int row = (c * kernel + ky) * kernel;
                            for (int kx = 0; kx < kernel; ++kx) {
                                int ix = ix0 + kx;
                                dst[column + row + kx] = rowInside && ix >= 0 && ix < inW
                                        ? src[sample + (iy * inW + ix) * inChannels + c]
                                        : 0.0f;
                            }
                        }
                    }
                }
            }
        }
    }

    /** The inverse of {@link #im2col}, accumulating because neighboring patches overlap. */
    private void col2im(float[] src, float[] dst, int m) {
        int patchRows = inChannels * kernel * kernel;
        int inSpatial = inH * inW;
        int outSpatial = outH * outW;
        Arrays.fill(dst, 0, m * inSpatial * inChannels, 0.0f);
        for (int s = 0; s < m; ++s) {
            int sample = s * inSpatial * inChannels;
            for (int oy = 0; oy < outH; ++oy) {
                for (int ox = 0; ox < outW; ++ox) {
                    int column = (s * outSpatial + oy * outW + ox) * patchRows;
                    int iy0 = oy * stride - pad;
                    int ix0 = ox * stride - pad;
                    for (int c = 0; c < inChannels; ++c) {
                        for (int ky = 0; ky < kernel; ++ky) {
                            int iy = iy0 + ky;
                            if (iy < 0 || iy >= inH) {
                                continue;
                            }
                            int row = (c * kernel + ky) * kernel;
                            for (int kx = 0; kx < kernel; ++kx) {
                                int ix = ix0 + kx;
                                if (ix >= 0 && ix < inW) {
                                    dst[sample + (iy * inW + ix) * inChannels + c] += src[column + row + kx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /** Writes the kernels if storing was enabled at construction time. */
    public void storeKernels() {
        if (storeKernelsAndBiases) {
            store(STORE_DIR + "w_" + name, kernels.value());
        }
    }

    /** Writes the biases if storing was enabled at construction time. */
    public void storeBiases() {
        if (storeKernelsAndBiases) {
            store(STORE_DIR + "b_" + name, biases.value());
        }
    }

    /**
     * Persists both the kernels and the biases of this layer if the
     * {@code storeKernelsAndBiases} flag was set at construction time.
     */
    @Override
    public void storeParameters() {
        storeKernels();
        storeBiases();
    }

    private MatrixF load(String path) {
        try (FileInputStream fis = new FileInputStream(path)) {
            return Matrices.deserializeF(fis);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private void store(String path, MatrixF matrix) {
        try {
            Files.createDirectories(Paths.get(STORE_DIR));
            try (FileOutputStream fos = new FileOutputStream(path)) {
                Matrices.serializeF(matrix, fos);
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
