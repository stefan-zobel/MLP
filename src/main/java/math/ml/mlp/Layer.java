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

import net.jamu.matrix.MatrixF;

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
     * @param learningRate the learning rate ({@code 0 < r < 1})
     * @return error gradients with respect to the input of this layer
     */
    MatrixF backward(MatrixF grads, float learningRate);

    void setMode(NetworkMode mode);
}
