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
import java.util.function.IntFunction;

import net.jamu.matrix.MatrixF;

public interface Loss extends Layer {

    void registerLossCallback(Consumer<MatrixF> callback);

    void registerAccuracyCallback(DoubleConsumer callback);

    void registerBatchExpectedValuesProvider(IntFunction<MatrixF> provider);

    // by default backward() for a Loss function does nothing and shouldn't be
    // called
    default MatrixF backward(MatrixF unused1, float unused2) {
        return null;
    }
}
