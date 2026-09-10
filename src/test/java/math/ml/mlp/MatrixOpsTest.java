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

import static math.ml.mlp.GradientCheck.input;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

class MatrixOpsTest {

    private static final int ROWS = 5;
    private static final int COLS = 3;

    @Test
    void mulRowsInplaceScalesEachRowByItsOwnFactor() {
        MatrixF a = input(ROWS, COLS, 601L);
        MatrixF original = a.copy();
        MatrixF v = input(ROWS, 1, 602L);

        assertSame(a, MatrixOps.mulRowsInplace(a, v));
        for (int c = 0; c < COLS; ++c) {
            for (int r = 0; r < ROWS; ++r) {
                assertEquals(original.getUnsafe(r, c) * v.getUnsafe(r, 0), a.getUnsafe(r, c), 1e-6f);
            }
        }
    }

    @Test
    void mulColumnsInplaceScalesEachColumnByItsOwnFactor() {
        MatrixF a = input(ROWS, COLS, 603L);
        MatrixF original = a.copy();
        MatrixF v = input(1, COLS, 604L);

        assertSame(a, MatrixOps.mulColumnsInplace(a, v));
        for (int c = 0; c < COLS; ++c) {
            for (int r = 0; r < ROWS; ++r) {
                assertEquals(original.getUnsafe(r, c) * v.getUnsafe(0, c), a.getUnsafe(r, c), 1e-6f);
            }
        }
    }

    @Test
    void addColumnsInplaceShiftsEachColumnByItsOwnOffset() {
        MatrixF a = input(ROWS, COLS, 605L);
        MatrixF original = a.copy();
        MatrixF v = input(1, COLS, 606L);

        assertSame(a, MatrixOps.addColumnsInplace(a, v));
        for (int c = 0; c < COLS; ++c) {
            for (int r = 0; r < ROWS; ++r) {
                assertEquals(original.getUnsafe(r, c) + v.getUnsafe(0, c), a.getUnsafe(r, c), 1e-6f);
            }
        }
    }

    @Test
    void divInplaceDividesElementWise() {
        MatrixF a = input(ROWS, COLS, 607L);
        MatrixF original = a.copy();
        // bounded away from zero so the reference division stays well conditioned
        MatrixF b = Matrices.randomUniformF(ROWS, COLS, 1.0f, 3.0f, 608L);

        assertSame(a, MatrixOps.divInplace(a, b));
        for (int c = 0; c < COLS; ++c) {
            for (int r = 0; r < ROWS; ++r) {
                assertEquals(original.getUnsafe(r, c) / b.getUnsafe(r, c), a.getUnsafe(r, c), 1e-5f);
            }
        }
    }

    @Test
    void theRowAndColumnHelpersRejectTheWrongOrientation() {
        MatrixF a = input(ROWS, COLS, 609L);
        // a row vector where a column vector is wanted, and vice versa
        assertThrows(IllegalArgumentException.class, () -> MatrixOps.mulRowsInplace(a, input(1, COLS, 610L)));
        assertThrows(IllegalArgumentException.class, () -> MatrixOps.mulColumnsInplace(a, input(ROWS, 1, 611L)));
        assertThrows(IllegalArgumentException.class, () -> MatrixOps.addColumnsInplace(a, input(ROWS, 1, 612L)));
    }

    @Test
    void divInplaceRejectsMismatchedDimensions() {
        MatrixF a = input(ROWS, COLS, 613L);
        assertThrows(IllegalArgumentException.class, () -> MatrixOps.divInplace(a, input(ROWS, COLS + 1, 614L)));
    }
}
