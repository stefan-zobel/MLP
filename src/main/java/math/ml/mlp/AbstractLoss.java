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

public class AbstractLoss extends AbstractLayer implements Loss {

    protected Consumer<MatrixF> lossCallback;
    protected DoubleConsumer accuracyCallback;
    protected MatrixF expectedValues;

    public AbstractLoss() {
    }

    @Override
    public void registerLossCallback(Consumer<MatrixF> callback) {
        lossCallback = callback;
    }

    @Override
    public void registerAccuracyCallback(DoubleConsumer callback) {
        accuracyCallback = callback;
    }

    @Override
    public void setExpectedValues(MatrixF expected) {
        expectedValues = expected;
    }

    /**
     * Returns the target values pushed in by {@link #setExpectedValues(MatrixF)}
     * and clears them, so that a second {@code forward()} cannot silently reuse
     * stale targets.
     *
     * @return the target values for the current batch
     * @throws IllegalStateException if no targets were supplied
     */
    protected MatrixF getExpectation() {
        MatrixF expected = expectedValues;
        if (expected == null) {
            throw new IllegalStateException(
                    "no expected values were set; a Loss cannot score predictions without targets");
        }
        expectedValues = null;
        return expected;
    }
}
