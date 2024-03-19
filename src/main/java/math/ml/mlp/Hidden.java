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

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

public class Hidden extends AbstractLayer {

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
        return load("./data/w_" + name);
    }

    private MatrixF loadBiases() {
        return load("./data/b_" + name);
    }

    public void storeWeights() {
        if (storeWeightsAndBiases) {
            store("w_" + name, weights);
        }
    }

    public void storeBiases() {
        if (storeWeightsAndBiases) {
            store("b_" + name, biases);
        }
    }

    private MatrixF load(String name) {
        try (FileInputStream fis = new FileInputStream(name)) {
            return Matrices.deserializeF(fis);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private void store(String name, MatrixF matrix) {
        try (FileOutputStream fos = new FileOutputStream(name)) {
            Matrices.serializeF(matrix, fos);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
