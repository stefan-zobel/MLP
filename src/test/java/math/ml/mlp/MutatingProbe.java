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

import java.util.Arrays;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

// Overwrites both of its arguments with NaN and returns a constant of its own, so a
// composite that fails to make a defensive copy is caught by a NaN reaching the
// identity path or a sibling branch. Dropout is the real layer that behaves this way.
final class MutatingProbe extends AbstractLayer {

    private final int rows;
    private final int cols;
    private final float constant;

    MutatingProbe(int rows, int cols, float constant) {
        this.rows = rows;
        this.cols = cols;
        this.constant = constant;
    }

    @Override
    public boolean mutatesInput() {
        return true;
    }

    @Override
    public boolean mutatesGradients() {
        return true;
    }

    @Override
    public MatrixF forward(MatrixF in) {
        Arrays.fill(in.getArrayUnsafe(), Float.NaN);
        return filled(constant);
    }

    @Override
    public MatrixF backward(MatrixF grads) {
        Arrays.fill(grads.getArrayUnsafe(), Float.NaN);
        return filled(0.0f);
    }

    private MatrixF filled(float value) {
        MatrixF m = Matrices.createF(rows, cols);
        Arrays.fill(m.getArrayUnsafe(), value);
        return m;
    }
}
