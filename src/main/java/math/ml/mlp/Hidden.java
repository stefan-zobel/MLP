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

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

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

    // j x i
    protected final MatrixF weights;
    // j x 1
    protected final MatrixF biases;
    protected final String name;
    protected final boolean storeWeightsAndBiases;

    public Hidden(int in, int out, String name) {
        this(in, out, name, false, false);
    }

    public Hidden(int in, int out, String name, boolean loadWeightsAndBiases, boolean storeWeightsAndBiases) {
        this.name = name;
        this.storeWeightsAndBiases = storeWeightsAndBiases;
        int i = in;
        int j = out;
        if (loadWeightsAndBiases) {
            weights = loadWeights();
            biases = loadBiases();
        } else {
            // Glorot uniform initialization
            float bound = (float) Math.sqrt(6.0 / (i + j));
            weights = Matrices.randomUniformF(j, i, -bound, bound);
            biases = Matrices.createF(j, 1);
        }
    }

    @Override
    public MatrixF forward(MatrixF input) {
        super.forward(input);
        // (j x i) * (i x m) + (j x m) = (j x m)
        return weights.times(input).addBroadcastedVectorInplace(biases);
    }

    // outputGrads : j x m
    @Override
    public MatrixF backward(MatrixF outputGrads, float learningRate) {
        if (mode == NetworkMode.INFER) {
            return null;
        }
        // (i x j) * (j x m) = (i x m)
        MatrixF inputErrJacobian = weights.transposedTimes(outputGrads);
        MatrixF avgWeightsGrad = outputGrads.timesTransposed(input).scaleInplace(1.0f / outputGrads.numColumns());
        input = null;
        // j x 1
        MatrixF avgBiasesGrad = Matrices.colsAverage(outputGrads);
        weights.addInplace(-learningRate, avgWeightsGrad);
        biases.addInplace(-learningRate, avgBiasesGrad);
        return inputErrJacobian;
    }

    private MatrixF loadWeights() {
        return load(LOAD_DIR + "w_" + name);
    }

    private MatrixF loadBiases() {
        return load(LOAD_DIR + "b_" + name);
    }

    public void storeWeights() {
        if (storeWeightsAndBiases) {
            store(STORE_DIR + "w_" + name, weights);
        }
    }

    public void storeBiases() {
        if (storeWeightsAndBiases) {
            store(STORE_DIR + "b_" + name, biases);
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
