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

import java.util.Objects;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

/**
 * A trainable tensor together with the gradient the last backward pass produced
 * for it. A layer owns its parameters and writes their gradients; an
 * {@link Optimizer} reads them and applies the update.
 */
public final class Parameter {

    private final MatrixF value;
    private final MatrixF grad;
    private final String name;
    private final boolean decayed;

    /**
     * Wraps a matrix as a trainable parameter and allocates its gradient buffer.
     *
     * @param name    a label for diagnostics; not a file name
     * @param value   the matrix to train, used as it is rather than copied
     * @param decayed whether weight decay applies to this parameter
     */
    public Parameter(String name, MatrixF value, boolean decayed) {
        this.name = Objects.requireNonNull(name, "name");
        this.value = Objects.requireNonNull(value, "value");
        this.grad = Matrices.sameDimF(value);
        this.decayed = decayed;
    }

    /**
     * The matrix being trained.
     *
     * @return the parameter value
     */
    public MatrixF value() {
        return value;
    }

    /**
     * The gradient of the loss with respect to {@link #value()}, averaged over the
     * batch. Every backward pass overwrites it; it is never accumulated.
     *
     * @return the gradient buffer, which has the shape of the value
     */
    public MatrixF grad() {
        return grad;
    }

    /**
     * A label for diagnostics.
     *
     * @return the name this parameter was created with
     */
    public String name() {
        return name;
    }

    /**
     * Whether weight decay applies. True for weight matrices, false for biases and
     * for the scale and shift of a normalization layer.
     *
     * @return {@code true} if an optimizer should decay this parameter
     */
    public boolean isDecayed() {
        return decayed;
    }

    @Override
    public String toString() {
        return "Parameter[" + name + " " + value.numRows() + "x" + value.numColumns() + "]";
    }
}
