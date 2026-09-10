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

import net.jamu.matrix.FFunction;
import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

/** An element-wise activation, given as a function and its derivative. */
public class Activation extends AbstractLayer {

    /** The activation itself. */
    protected final FFunction fun;
    /** Its derivative, applied to the cached pre-activation value. */
    protected final FFunction deriv;

    /**
     * Creates an activation layer.
     *
     * @param fun   the activation function
     * @param deriv its derivative with respect to the pre-activation value
     */
    public Activation(FFunction fun, FFunction deriv) {
        this.fun = fun;
        this.deriv = deriv;
    }

    @Override
    public MatrixF forward(MatrixF input) {
        // j x m
        super.forward(input);
        float[] in = input.getArrayUnsafe();
        MatrixF output = Matrices.sameDimF(input);
        applyForward(in, output.getArrayUnsafe(), 0, in.length);
        return output;
    }

    // outputGrads : j x m
    @Override
    public MatrixF backward(MatrixF outputGrads, float unused) {
        if (mode == NetworkMode.INFER) {
            return null;
        }
        // (j x m) o (j x m)
        checkSameDimension(outputGrads);
        float[] grads = outputGrads.getArrayUnsafe();
        MatrixF out = Matrices.sameDimF(outputGrads);
        applyBackward(input.getArrayUnsafe(), grads, out.getArrayUnsafe(), 0, grads.length);
        input = null;
        return out;
    }

    /**
     * Writes the activation of {@code in[from, to)} into {@code out}.
     *
     * <p>The arrays are the column-major backing arrays of the matrices, so a range
     * of columns is the contiguous range {@code [firstColumn * rows, ...)}. Every
     * subclass overrides this with its own loop: routing all of them through one
     * shared {@link FFunction} call site is what costs the JIT its inlining.
     *
     * @param in   the pre-activation values
     * @param out  where to write the result
     * @param from index of the first element to touch
     * @param to   index one past the last element to touch
     */
    void applyForward(float[] in, float[] out, int from, int to) {
        for (int i = from; i < to; ++i) {
            out[i] = fun.apply(in[i]);
        }
    }

    /**
     * Writes {@code grads * deriv(preAct)} over {@code [from, to)} into {@code out}.
     *
     * @param preAct the cached pre-activation values
     * @param grads  the gradients with respect to this layer's output
     * @param out    where to write the result
     * @param from   index of the first element to touch
     * @param to     index one past the last element to touch
     */
    void applyBackward(float[] preAct, float[] grads, float[] out, int from, int to) {
        for (int i = from; i < to; ++i) {
            out[i] = grads[i] * deriv.apply(preAct[i]);
        }
    }

    /** The matrix API checks this for us; a raw loop has to do it itself. */
    private void checkSameDimension(MatrixF outputGrads) {
        if (input.numRows() != outputGrads.numRows() || input.numColumns() != outputGrads.numColumns()) {
            throw new IllegalArgumentException("dimension mismatch: forward saw " + input.numRows() + " x "
                    + input.numColumns() + ", backward got " + outputGrads.numRows() + " x "
                    + outputGrads.numColumns());
        }
    }
}
