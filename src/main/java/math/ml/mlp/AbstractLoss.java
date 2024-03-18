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

public class AbstractLoss extends AbstractLayer implements Loss {

    protected Consumer<MatrixF> lossCallback;
    protected DoubleConsumer accuracyCallback;
    protected IntFunction<MatrixF> expectedBatchResultsCallback;

    int batchNumber = 0;

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
    public void registerBatchExpectedValuesProvider(IntFunction<MatrixF> provider) {
        expectedBatchResultsCallback = provider;
    }

    /**
     * Get and return the expected values for this batch by means of the current
     * batch number. Increases the batch number by one if that succeeds, otherwise
     * returns {@code null}.
     * 
     * @return the expected values for the current batch or {@code null} if the
     *         retrieval doesn't succeed
     */
    public MatrixF getExpectation() {
        MatrixF expected = null;
        if (expectedBatchResultsCallback == null
                || (expected = expectedBatchResultsCallback.apply(batchNumber)) == null) {
            // there is nothing a Loss function can do without knowing what the expected
            // values for the predictions are
            return null;
        }
        ++batchNumber;
        return expected;
    }
}
