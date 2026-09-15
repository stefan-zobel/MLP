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

import java.io.IOException;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

/** A fully connected layer, {@code y = W x + b}. */
public class Hidden extends AbstractLayer {

    /** The weight matrix, out x in. */
    protected final Parameter weights;
    /** The bias column, out x 1. */
    protected final Parameter biases;
    /** Names the bundle entries of this layer; without a name it has none. */
    protected final String name;

    /**
     * Creates a layer with Glorot initialization from an unseeded draw.
     *
     * @param in   number of input features
     * @param out  number of output features
     * @param name names the bundle entries {@code <name>/weights} and {@code <name>/biases}
     */
    public Hidden(int in, int out, String name) {
        this(in, out, name, ThreadLocalRandom.current().nextLong());
    }

    /**
     * Creates a layer whose weight initialization is reproducible.
     *
     * @param in   number of input features
     * @param out  number of output features
     * @param name names the bundle entries {@code <name>/weights} and {@code <name>/biases}
     * @param seed seed for the weight draw
     */
    public Hidden(int in, int out, String name, long seed) {
        this(in, out, name, Init.GLOROT, seed);
    }

    /**
     * Creates a layer with an explicit initialization scheme; use {@link Init#HE}
     * when a ReLU or GELU follows.
     *
     * @param in   number of input features
     * @param out  number of output features
     * @param name names the bundle entries {@code <name>/weights} and {@code <name>/biases}
     * @param init the weight initialization scheme
     * @param seed seed for the weight draw
     */
    public Hidden(int in, int out, String name, Init init, long seed) {
        this.name = name;
        float bound = init.bound(in, out);
        // weight decay applies to the matrix but not to the bias, the standard rule
        weights = new Parameter("weights", Matrices.randomUniformF(out, in, -bound, bound, seed), true);
        biases = new Parameter("biases", Matrices.createF(out, 1), false);
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

    @Override
    public void writeParameters(ParameterSink sink) throws IOException {
        ParameterStore.requireName(name, this);
        ParameterStore.write(sink, name + "/weights", weights.value());
        ParameterStore.write(sink, name + "/biases", biases.value());
    }

    @Override
    public void readParameters(ParameterSource source) throws IOException {
        ParameterStore.requireName(name, this);
        ParameterStore.read(source, name + "/weights", weights.value());
        ParameterStore.read(source, name + "/biases", biases.value());
    }
}
