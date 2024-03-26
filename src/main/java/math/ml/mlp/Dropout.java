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

import java.util.BitSet;

import math.rng.XorShiftRot256StarStar;
import net.jamu.matrix.MatrixF;

public class Dropout extends AbstractLayer {

    private final float dropoutRate;
    private final float scalingFactor;
    private BitSet mask = new BitSet(0);

    public Dropout(float dropoutRate) {
        this.dropoutRate = dropoutRate;
        this.scalingFactor = 1.0f / (1.0f - dropoutRate);
    }

    // input: j x m
    @Override
    public MatrixF forward(MatrixF input) {
        if (mode == NetworkMode.INFER || dropoutRate <= 0.0f) {
            return input;
        }
        int inputSize = input.numRows() * input.numColumns();
        if (mask.size() != inputSize) {
            mask = new BitSet(inputSize);
        } else {
            mask.clear();
        }
        for (int col = 0; col < input.numColumns(); ++col) {
            for (int row = 0; row < input.numRows(); ++row) {
                if (dropout()) {
                    mask.set(col * row);
                    input.setUnsafe(row, col, 0.0f);
                } else {
                    input.setUnsafe(row, col, scalingFactor * input.getUnsafe(row, col));
                }
            }
        }
        return input;
    }

    @Override
    public MatrixF backward(MatrixF grads, float unused) {
        if (mode == NetworkMode.INFER) {
            return null;
        }
        for (int col = 0; col < grads.numColumns(); ++col) {
            for (int row = 0; row < grads.numRows(); ++row) {
                if (mask.get(row * col)) {
                    grads.setUnsafe(row, col, 0.0f);
                } else {
                    grads.setUnsafe(row, col, scalingFactor * grads.getUnsafe(row, col));
                }
            }
        }
        return grads;
    }

    private boolean dropout() {
        return XorShiftRot256StarStar.getDefault().nextFloat() < dropoutRate;
    }
}
