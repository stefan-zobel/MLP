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
import java.util.concurrent.ThreadLocalRandom;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

/**
 * Adds one learned vector per sequence position to a layout of {@code seqLen} columns per
 * sample, the same positions repeating in every block. Attention is indifferent to the
 * order of its tokens, so without this a sequence is a set.
 */
public class PositionalEncoding extends AbstractLayer {

    /** The position table, dModel x seqLen, one column per position. */
    protected final Parameter positions;
    /** Identifies the parameter file of this layer. */
    protected final String name;
    /** Whether {@link #storeParameters()} writes anything. */
    protected final boolean storePositions;

    private final int dModel;
    private final int seqLen;

    private MatrixF output;
    /** The batch the buffer above was sized for, or -1 before the first pass. */
    private int batch = -1;

    /**
     * Creates a position table from an unseeded draw.
     *
     * @param dModel features per token
     * @param seqLen tokens per sample
     * @param name   identifies the parameter file {@code w_<name>}
     */
    public PositionalEncoding(int dModel, int seqLen, String name) {
        this(dModel, seqLen, name, false, false, ThreadLocalRandom.current().nextLong());
    }

    /**
     * Creates a position table whose initialization is reproducible.
     *
     * @param dModel features per token
     * @param seqLen tokens per sample
     * @param name   identifies the parameter file {@code w_<name>}
     * @param seed   seed for the draw
     */
    public PositionalEncoding(int dModel, int seqLen, String name, long seed) {
        this(dModel, seqLen, name, false, false, seed);
    }

    /**
     * The overloads without the flags neither load nor store.
     *
     * @param dModel         features per token
     * @param seqLen         tokens per sample
     * @param name           identifies the parameter file {@code w_<name>}
     * @param loadPositions  read the table from {@code ./data/} at construction
     * @param storePositions let {@link #storeParameters()} write to {@code ./checkpoints/}
     * @param seed           seed for the draw, unused when the table is loaded
     */
    public PositionalEncoding(int dModel, int seqLen, String name, boolean loadPositions, boolean storePositions,
            long seed) {
        if (dModel <= 0 || seqLen <= 0) {
            throw new IllegalArgumentException("dModel " + dModel + " and seqLen " + seqLen + " must both be positive");
        }
        this.dModel = dModel;
        this.seqLen = seqLen;
        this.name = name;
        this.storePositions = storePositions;
        MatrixF p;
        if (loadPositions) {
            p = load(ParameterStore.LOAD_DIR + "w_" + name);
        } else {
            // the table is added to the activations rather than multiplied with them, so the
            // fan-in rules of Init do not apply; this keeps it small against a unit-scale input
            float bound = 1.0f / (float) Math.sqrt(dModel);
            p = Matrices.randomUniformF(dModel, seqLen, -bound, bound, seed);
        }
        // no weight decay, for the same reason a bias carries none
        positions = new Parameter("positions", p, false);
    }

    /** The position table. */
    @Override
    public List<Parameter> parameters() {
        return List.of(positions);
    }

    @Override
    public MatrixF forward(MatrixF in) {
        if (in.numRows() != dModel) {
            throw new IllegalArgumentException("expected " + dModel + " features, got " + in.numRows());
        }
        int cols = in.numColumns();
        if (cols % seqLen != 0) {
            throw new IllegalArgumentException(cols + " columns is not a whole number of sequences of " + seqLen);
        }
        int m = cols / seqLen;
        ensureBuffers(m);
        float[] src = in.getArrayUnsafe();
        float[] table = positions.value().getArrayUnsafe();
        float[] dst = output.getArrayUnsafe();
        for (int c = 0; c < cols; ++c) {
            int from = c * dModel;
            int pos = (c % seqLen) * dModel;
            for (int r = 0; r < dModel; ++r) {
                dst[from + r] = src[from + r] + table[pos + r];
            }
        }
        return output;
    }

    @Override
    public MatrixF backward(MatrixF outputGrads) {
        if (mode == NetworkMode.INFER) {
            return null;
        }
        int cols = outputGrads.numColumns();
        int m = cols / seqLen;
        float[] src = outputGrads.getArrayUnsafe();
        float[] dst = positions.grad().getArrayUnsafe();
        Arrays.fill(dst, 0.0f);
        for (int c = 0; c < cols; ++c) {
            int from = c * dModel;
            int pos = (c % seqLen) * dModel;
            for (int r = 0; r < dModel; ++r) {
                dst[pos + r] += src[from + r];
            }
        }
        // one position occurs once per sample, so the mean over the batch divides by the
        // number of samples and not by the number of columns
        positions.grad().scaleInplace(1.0f / m);
        // adding a constant leaves the derivative with respect to the input the identity
        return outputGrads;
    }

    private void ensureBuffers(int m) {
        if (batch == m) {
            return;
        }
        output = Matrices.createF(dModel, m * seqLen);
        batch = m;
    }

    /**
     * Persists the position table if storing was enabled at construction time.
     */
    @Override
    public void storeParameters() {
        if (storePositions) {
            try {
                Files.createDirectories(Paths.get(ParameterStore.STORE_DIR));
                try (FileOutputStream fos = new FileOutputStream(ParameterStore.STORE_DIR + "w_" + name)) {
                    Matrices.serializeF(positions.value(), fos);
                }
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
    }

    private MatrixF load(String path) {
        try (FileInputStream fis = new FileInputStream(path)) {
            return Matrices.deserializeF(fis);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
