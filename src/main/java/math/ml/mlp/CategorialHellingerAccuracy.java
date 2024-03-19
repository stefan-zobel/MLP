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

import math.dl.Hellinger;
import net.jamu.matrix.MatrixF;

/**
 * Calculation of the accuracy of a categorial classifier.
 */
public final class CategorialHellingerAccuracy {

    /**
     * Calculation of the accuracy of a categorial classifier.
     * 
     * @param pred
     * @param expect
     * @return
     */
    public static double computeAccuracy(MatrixF pred, MatrixF expect) {
        double accuracy = 0.0;
        int length = pred.numRows();
        float[] a = pred.getArrayUnsafe();
        float[] b = expect.getArrayUnsafe();
        for (int off = 0; off < length * pred.numColumns(); off += length) {
            accuracy += Hellinger.similarityF(length, a, off, b);
        }
        return accuracy / pred.numColumns();
    }

    private CategorialHellingerAccuracy() {
        throw new AssertionError();
    }
}
