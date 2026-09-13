/*
 * Copyright 2024 Stefan Zobel
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
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

/** A fully connected layer, {@code y = W x + b}. */
public class Hidden extends AbstractLayer {

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

    /** The weight matrix, out x in. */
    protected final Parameter weights;
    /** The bias column, out x 1. */
    protected final Parameter biases;
    /** Identifies the parameter files of this layer. */
    protected final String name;
    /** Whether {@link #storeParameters()} writes anything. */
    protected final boolean storeWeightsAndBiases;

    /**
     * Creates a layer with Glorot initialization from an unseeded draw.
     *
     * @param in   number of input features
     * @param out  number of output features
     * @param name identifies the parameter files {@code w_<name>} and {@code b_<name>}
     */
    public Hidden(int in, int out, String name) {
        this(in, out, name, false, false, ThreadLocalRandom.current().nextLong());
    }

    /**
     * Creates a layer whose weight initialization is reproducible.
     *
     * @param in   number of input features
     * @param out  number of output features
     * @param name identifies the parameter files {@code w_<name>} and {@code b_<name>}
     * @param seed seed for the weight draw
     */
    public Hidden(int in, int out, String name, long seed) {
        this(in, out, name, false, false, Init.GLOROT, seed);
    }

    /**
     * Creates a layer with an explicit initialization scheme; use {@link Init#HE}
     * when a ReLU or GELU follows.
     *
     * @param in   number of input features
     * @param out  number of output features
     * @param name identifies the parameter files {@code w_<name>} and {@code b_<name>}
     * @param init the weight initialization scheme
     * @param seed seed for the weight draw
     */
    public Hidden(int in, int out, String name, Init init, long seed) {
        this(in, out, name, false, false, init, seed);
    }

    /**
     * Creates a layer with Glorot initialization from an unseeded draw.
     *
     * @param in                     number of input features
     * @param out                    number of output features
     * @param name                   identifies the parameter files {@code w_<name>} and {@code b_<name>}
     * @param loadWeightsAndBiases   read the parameters from {@code ./data/} at construction
     * @param storeWeightsAndBiases  let {@link #storeParameters()} write to {@code ./checkpoints/}
     */
    public Hidden(int in, int out, String name, boolean loadWeightsAndBiases, boolean storeWeightsAndBiases) {
        this(in, out, name, loadWeightsAndBiases, storeWeightsAndBiases, ThreadLocalRandom.current().nextLong());
    }

    /**
     * The overloads without a {@code seed} draw one, so a run is reproducible only
     * if the seed is passed in.
     *
     * @param in                     number of input features
     * @param out                    number of output features
     * @param name                   identifies the parameter files {@code w_<name>} and {@code b_<name>}
     * @param loadWeightsAndBiases   read the parameters from {@code ./data/} at construction
     * @param storeWeightsAndBiases  let {@link #storeParameters()} write to {@code ./checkpoints/}
     * @param seed                   seed for the weight draw, unused when the parameters are loaded
     */
    public Hidden(int in, int out, String name, boolean loadWeightsAndBiases, boolean storeWeightsAndBiases,
            long seed) {
        this(in, out, name, loadWeightsAndBiases, storeWeightsAndBiases, Init.GLOROT, seed);
    }

    /**
     * The overloads without an {@code init} use {@link Init#GLOROT}.
     *
     * @param in                     number of input features
     * @param out                    number of output features
     * @param name                   identifies the parameter files {@code w_<name>} and {@code b_<name>}
     * @param loadWeightsAndBiases   read the parameters from {@code ./data/} at construction
     * @param storeWeightsAndBiases  let {@link #storeParameters()} write to {@code ./checkpoints/}
     * @param init                   the weight initialization scheme
     * @param seed                   seed for the weight draw, unused when the parameters are loaded
     */
    public Hidden(int in, int out, String name, boolean loadWeightsAndBiases, boolean storeWeightsAndBiases,
            Init init, long seed) {
        this.name = name;
        this.storeWeightsAndBiases = storeWeightsAndBiases;
        int i = in;
        int j = out;
        MatrixF w;
        MatrixF b;
        if (loadWeightsAndBiases) {
            w = loadWeights();
            b = loadBiases();
        } else {
            float bound = init.bound(i, j);
            w = Matrices.randomUniformF(j, i, -bound, bound, seed);
            b = Matrices.createF(j, 1);
        }
        // weight decay applies to the matrix but not to the bias, the standard rule
        weights = new Parameter("weights", w, true);
        biases = new Parameter("biases", b, false);
    }

    /** The weight matrix and the bias column. */
    @Override
    public List<Parameter> parameters() {
        return List.of(weights, biases);
    }

    @Override
    public MatrixF forward(MatrixF input) {
        super.forward(input);
        // (j x i) * (i x m) + (j x m) = (j x m)
        return weights.value().times(input).addBroadcastedVectorInplace(biases.value());
    }

    // outputGrads : j x m
    @Override
    public MatrixF backward(MatrixF outputGrads) {
        if (mode == NetworkMode.INFER) {
            return null;
        }
        // (i x j) * (j x m) = (i x m)
        MatrixF inputErrJacobian = weights.value().transposedTimes(outputGrads);
        // timesTransposed() is transBmult() into a freshly created matrix, so writing into
        // the parameter's own buffer is the same call without the j x i allocation. The
        // result is not bit-identical to the allocating form, because MKL's sgemm depends
        // on the alignment of its destination; a fixed buffer is in fact the more stable
        // of the two, since its alignment no longer varies with the allocation history.
        outputGrads.transBmult(input, weights.grad()).scaleInplace(1.0f / outputGrads.numColumns());
        input = null;
        // j x 1, and jamu has no colsAverage() with a destination; too small to matter
        biases.grad().setInplace(Matrices.colsAverage(outputGrads));
        return inputErrJacobian;
    }

    private MatrixF loadWeights() {
        return load(LOAD_DIR + "w_" + name);
    }

    private MatrixF loadBiases() {
        return load(LOAD_DIR + "b_" + name);
    }

    /** Writes the weights if storing was enabled at construction time. */
    public void storeWeights() {
        if (storeWeightsAndBiases) {
            store(STORE_DIR + "w_" + name, weights.value());
        }
    }

    /** Writes the biases if storing was enabled at construction time. */
    public void storeBiases() {
        if (storeWeightsAndBiases) {
            store(STORE_DIR + "b_" + name, biases.value());
        }
    }

    /**
     * Persists both the weights and the biases of this layer if the
     * {@code storeWeightsAndBiases} flag was set at construction time.
     */
    @Override
    public void storeParameters() {
        storeWeights();
        storeBiases();
    }

    private MatrixF load(String name) {
        try (FileInputStream fis = new FileInputStream(name)) {
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
