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

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

/** A {@link Network} that can also be trained. */
public interface TrainableNetwork extends Network {

    /**
     * Trains one batch. Inputs and targets are passed together so that they
     * cannot come apart.
     *
     * @param input        the batch to train on, one sample per column
     * @param expected     the target values for exactly that batch
     * @param learningRate the learning rate ({@code 0 < r < 1})
     * @return this network
     */
    Network train(MatrixF input, MatrixF expected, float learningRate);

    /**
     * Appends a layer. The default is a no-op, because not every network is built
     * layer by layer.
     *
     * @param layer the layer to append
     * @return this network
     */
    default Network add(Layer layer) {
        // not every Network needs to be constructed layer by layer
        return this;
    }

    /**
     * Called once per batch with the per-sample losses; the default prints them.
     *
     * @param losses the losses of the batch just trained
     */
    default void onLossComputationCompleted(MatrixF losses) {
        System.out.println("Avg. loss: " + Matrices.colsAverage(losses).toScalar());
    }

    /**
     * Called once per batch with the accuracy; the default prints it.
     *
     * @param accuracy the accuracy of the batch just trained
     */
    default void onAccuracyComputationCompleted(double accuracy) {
        System.out.println("Accuracy: " + accuracy);
    }
}
