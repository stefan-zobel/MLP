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

import java.util.function.Consumer;
import java.util.function.DoubleConsumer;

import net.jamu.matrix.MatrixF;

/** The final layer of a network: scores a prediction against the targets. */
public interface Loss extends Layer {

    /**
     * Registers a callback that receives the per-sample losses of each batch.
     *
     * @param callback the callback, or {@code null} to remove it
     */
    void registerLossCallback(Consumer<MatrixF> callback);

    /**
     * Registers a callback that receives the accuracy of each batch.
     *
     * @param callback the callback, or {@code null} to remove it
     */
    void registerAccuracyCallback(DoubleConsumer callback);

    /**
     * Supplies the target values for the next {@code forward()}. The caller
     * passes inputs and targets together, so the two cannot drift apart.
     *
     * @param expected the target values for the batch about to be trained
     */
    void setExpectedValues(MatrixF expected);

    // by default backward() for a Loss function does nothing and shouldn't be
    // called
    default MatrixF backward(MatrixF unused1) {
        return null;
    }

    /**
     * Whether this loss also yields a usable prediction in
     * {@link NetworkMode#INFER} mode, which the fused losses do because they
     * apply the output activation themselves.
     *
     * @return {@code true} if this loss produces a prediction in INFER mode
     */
    default boolean producesPredictionInInferMode() {
        return false;
    }
}
