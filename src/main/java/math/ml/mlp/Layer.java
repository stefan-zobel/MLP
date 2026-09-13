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

import java.util.List;

import net.jamu.matrix.MatrixF;

/** One stage of a network: a forward pass, a backward pass and a mode. */
public interface Layer {

    /**
     * Forward pass.
     * 
     * @param input the forward input into this layer
     * @return the forward output of this layer
     */
    MatrixF forward(MatrixF input);

    /**
     * Backward pass.
     * 
     * @param grads error gradients with respect to the output of this layer
     * @return error gradients with respect to the input of this layer
     */
    MatrixF backward(MatrixF grads);

    /**
     * Switches between training and inference.
     *
     * @param mode the mode to run in
     */
    void setMode(NetworkMode mode);

    /**
     * Whether {@link #forward(MatrixF)} may overwrite the matrix it is handed.
     * Composite layers copy a shared input before passing it on if any of their
     * sub-layers says yes.
     *
     * @return {@code true} if the forward pass writes into its argument
     */
    default boolean mutatesInput() {
        return false;
    }

    /**
     * Whether {@link #backward(MatrixF)} may overwrite the matrix it is
     * handed. Neither predicate says anything about the matrix a layer
     * <em>returns</em>, which may still be one of its own buffers.
     *
     * @return {@code true} if the backward pass writes into its argument
     */
    default boolean mutatesGradients() {
        return false;
    }

    /**
     * The trainable parameters of this layer, for an {@link Optimizer} to update.
     * Composite layers concatenate those of their sub-layers, without removing
     * duplicates, exactly as {@link #storeParameters()} visits them.
     *
     * @return the parameters, empty by default
     */
    default List<Parameter> parameters() {
        return List.of();
    }

    /**
     * Persists the trainable parameters of this layer (e.g. weights and biases).
     * The default implementation is a no-op; layers with storable parameters
     * should override this method. Composite layers (e.g.
     * {@link ParallelBranches}) should delegate to all their sub-layers.
     */
    default void storeParameters() {
        // no-op by default
    }
}
