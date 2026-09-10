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

import net.jamu.matrix.MatrixF;

/**
 * The two element-wise operations Jamu does not offer, kept in one place so that
 * the layers themselves stay free of index arithmetic.
 */
final class MatrixOps {

    /**
     * Multiplies every column of {@code a} element-wise by the column vector
     * {@code v}, the multiplicative counterpart of
     * {@link MatrixF#addBroadcastedVectorInplace(MatrixF)}.
     *
     * @param a the matrix to scale per feature, modified in place
     * @param v a column vector with one entry per row of {@code a}
     * @return {@code a}
     */
    static MatrixF mulRowsInplace(MatrixF a, MatrixF v) {
        int rows = a.numRows();
        if (v.numRows() != rows || v.numColumns() != 1) {
            throw new IllegalArgumentException(
                    "expected a " + rows + " x 1 column vector, got " + v.numRows() + " x " + v.numColumns());
        }
        float[] x = a.getArrayUnsafe();
        float[] scale = v.getArrayUnsafe();
        int cols = a.numColumns();
        for (int c = 0, off = 0; c < cols; ++c, off += rows) {
            for (int r = 0; r < rows; ++r) {
                x[off + r] *= scale[r];
            }
        }
        return a;
    }

    /**
     * Adds {@code v[0, c]} to every element of column {@code c}, the row-vector
     * counterpart of {@link MatrixF#addBroadcastedVectorInplace(MatrixF)}, which
     * broadcasts column vectors only.
     *
     * @param a the matrix to shift per sample, modified in place
     * @param v a row vector with one entry per column of {@code a}
     * @return {@code a}
     */
    static MatrixF addColumnsInplace(MatrixF a, MatrixF v) {
        int cols = checkRowVector(a, v);
        int rows = a.numRows();
        float[] x = a.getArrayUnsafe();
        float[] shift = v.getArrayUnsafe();
        for (int c = 0, off = 0; c < cols; ++c, off += rows) {
            float s = shift[c];
            for (int r = 0; r < rows; ++r) {
                x[off + r] += s;
            }
        }
        return a;
    }

    /**
     * Multiplies every element of column {@code c} by {@code v[0, c]}.
     *
     * @param a the matrix to scale per sample, modified in place
     * @param v a row vector with one entry per column of {@code a}
     * @return {@code a}
     */
    static MatrixF mulColumnsInplace(MatrixF a, MatrixF v) {
        int cols = checkRowVector(a, v);
        int rows = a.numRows();
        float[] x = a.getArrayUnsafe();
        float[] scale = v.getArrayUnsafe();
        for (int c = 0, off = 0; c < cols; ++c, off += rows) {
            float s = scale[c];
            for (int r = 0; r < rows; ++r) {
                x[off + r] *= s;
            }
        }
        return a;
    }

    private static int checkRowVector(MatrixF a, MatrixF v) {
        int cols = a.numColumns();
        if (v.numRows() != 1 || v.numColumns() != cols) {
            throw new IllegalArgumentException(
                    "expected a 1 x " + cols + " row vector, got " + v.numRows() + " x " + v.numColumns());
        }
        return cols;
    }

    /**
     * Divides {@code a} by {@code b} element-wise.
     *
     * @param a the dividend, modified in place
     * @param b the divisor, same dimensions as {@code a}
     * @return {@code a}
     */
    static MatrixF divInplace(MatrixF a, MatrixF b) {
        if (a.numRows() != b.numRows() || a.numColumns() != b.numColumns()) {
            throw new IllegalArgumentException("dimension mismatch: " + a.numRows() + " x " + a.numColumns() + " and "
                    + b.numRows() + " x " + b.numColumns());
        }
        float[] x = a.getArrayUnsafe();
        float[] y = b.getArrayUnsafe();
        for (int i = 0; i < x.length; ++i) {
            x[i] /= y[i];
        }
        return a;
    }

    private MatrixOps() {
        throw new AssertionError();
    }
}
