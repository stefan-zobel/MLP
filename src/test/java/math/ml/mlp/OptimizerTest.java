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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

class OptimizerTest {

    private static final float RATE = 0.05f;

    @Test
    void theGradientBufferHasTheShapeOfTheValueAndStartsAtZero() {
        MatrixF value = Matrices.randomUniformF(6, 5, -1.0f, 1.0f, 71L);
        Parameter p = new Parameter("w", value, true);

        assertSame(value, p.value(), "the value must be the very matrix that was handed in");
        assertNotSame(p.value(), p.grad());
        assertEquals(6, p.grad().numRows());
        assertEquals(5, p.grad().numColumns());
        for (float g : p.grad().getArrayUnsafe()) {
            assertEquals(0.0f, g);
        }
    }

    @Test
    void sgdReproducesTheMatrixApiUpdateBitForBit() {
        // what the three layers did in their own backward() before the optimizer existed
        MatrixF expected = Matrices.randomUniformF(6, 5, -1.0f, 1.0f, 72L);
        MatrixF gradient = Matrices.randomUniformF(6, 5, -1.0f, 1.0f, 73L);
        Parameter p = new Parameter("w", expected.copy(), true);
        p.grad().setInplace(gradient);

        expected.addInplace(-RATE, gradient);

        Sgd sgd = new Sgd(RATE);
        sgd.add(p);
        sgd.step();

        assertBitwise(expected, p.value());
    }

    @Test
    void sgdReproducesTheMatrixApiOverManySteps() {
        MatrixF expected = Matrices.randomUniformF(4, 3, -1.0f, 1.0f, 74L);
        Parameter p = new Parameter("w", expected.copy(), true);
        Sgd sgd = new Sgd(RATE);
        sgd.add(p);

        for (int step = 0; step < 20; ++step) {
            MatrixF gradient = Matrices.randomUniformF(4, 3, -1.0f, 1.0f, 80L + step);
            p.grad().setInplace(gradient);
            expected.addInplace(-RATE, gradient);
            sgd.step();
            assertBitwise(expected, p.value());
        }
        assertEquals(20, sgd.steps());
    }

    @Test
    void aRateOfZeroTouchesNothing() {
        // addInplace skips its loop at alpha zero, so a zero rate must not write at all:
        // it would otherwise turn -0.0f into +0.0f and a non-finite gradient into a
        // non-finite parameter. GradientCheck relies on this.
        Parameter p = new Parameter("w", Matrices.createF(1, 4), false);
        float[] value = p.value().getArrayUnsafe();
        value[0] = -0.0f;
        value[1] = 0.0f;
        value[2] = 1.5f;
        value[3] = -2.5f;
        float[] gradient = p.grad().getArrayUnsafe();
        gradient[0] = -1.0f;
        gradient[1] = Float.NaN;
        gradient[2] = Float.POSITIVE_INFINITY;
        gradient[3] = Float.NaN;
        int[] before = bits(value);

        Sgd sgd = new Sgd(0.0f);
        sgd.add(p);
        sgd.step();

        assertEquals(1, sgd.steps(), "a zero rate still consumes a step");
        for (int i = 0; i < before.length; ++i) {
            assertEquals(before[i], Float.floatToRawIntBits(value[i]), "element " + i);
        }
    }

    @Test
    void negativeZeroIsAZeroRateToo() {
        Parameter p = new Parameter("w", Matrices.createF(1, 1), false);
        p.value().getArrayUnsafe()[0] = -0.0f;
        p.grad().getArrayUnsafe()[0] = 1.0f;

        Sgd sgd = new Sgd(-0.0f);
        sgd.add(p);
        sgd.step();

        assertEquals(Float.floatToRawIntBits(-0.0f), Float.floatToRawIntBits(p.value().getArrayUnsafe()[0]));
    }

    @Test
    void theSameParameterCannotBeRegisteredTwice() {
        // a layer instance shared between two composites would otherwise be updated
        // twice per step; ParallelBranches cannot run one anyway, so this fails early
        Parameter p = new Parameter("w", Matrices.createF(2, 2), true);
        Sgd sgd = new Sgd(RATE);
        sgd.add(p);

        assertThrows(IllegalArgumentException.class, () -> sgd.add(p));
    }

    @Test
    void twoParametersOfTheSameShapeAndNameAreStillTwo() {
        Sgd sgd = new Sgd(RATE);
        sgd.add(new Parameter("w", Matrices.createF(2, 2), true));
        sgd.add(new Parameter("w", Matrices.createF(2, 2), true));
    }

    @Test
    void theScheduleIsAskedOncePerStepStartingAtOne() {
        List<Integer> seen = new ArrayList<>();
        Sgd sgd = new Sgd(step -> {
            seen.add(step);
            return 0.0f;
        });
        sgd.add(new Parameter("w", Matrices.createF(2, 2), true));

        sgd.step();
        sgd.step();
        sgd.step();

        assertEquals(List.of(1, 2, 3), seen);
    }

    @Test
    void theScheduleDrivesTheRate() {
        float[] rates = { 0.125f, 0.25f, 0.5f };
        Parameter p = new Parameter("w", Matrices.createF(1, 1), true);
        p.grad().getArrayUnsafe()[0] = 1.0f;
        Sgd sgd = new Sgd(step -> rates[step - 1]);
        sgd.add(p);

        sgd.step();
        assertEquals(-0.125f, p.value().getArrayUnsafe()[0]);
        sgd.step();
        assertEquals(-0.375f, p.value().getArrayUnsafe()[0]);
        sgd.step();
        assertEquals(-0.875f, p.value().getArrayUnsafe()[0]);
    }

    @Test
    void anOptimizerWithoutParametersStepsWithoutComplaining() {
        Sgd sgd = new Sgd(RATE);
        sgd.step();
        assertEquals(1, sgd.steps());
    }

    @Test
    void sgdWithoutMomentumIsBitIdenticalToPlainSgd() {
        MatrixF start = Matrices.randomUniformF(5, 4, -1.0f, 1.0f, 91L);
        Parameter plain = new Parameter("w", start.copy(), true);
        Parameter zeroMomentum = new Parameter("w", start.copy(), true);
        Sgd a = new Sgd(RATE);
        Sgd b = new Sgd(RATE, 0.0f);
        a.add(plain);
        b.add(zeroMomentum);

        for (int step = 0; step < 10; ++step) {
            MatrixF g = Matrices.randomUniformF(5, 4, -1.0f, 1.0f, 92L + step);
            plain.grad().setInplace(g);
            zeroMomentum.grad().setInplace(g);
            a.step();
            b.step();
        }
        assertBitwise(plain.value(), zeroMomentum.value());
    }

    @Test
    void sgdMomentumFollowsTheRecurrence() {
        // b = momentum * b + grad, then p -= rate * b
        Parameter p = new Parameter("w", Matrices.createF(1, 1), true);
        p.grad().getArrayUnsafe()[0] = 1.0f;
        Sgd sgd = new Sgd(0.1f, 0.9f);
        sgd.add(p);

        float b = 0.0f;
        float expected = 0.0f;
        for (int step = 0; step < 3; ++step) {
            b = 0.9f * b + 1.0f;
            expected -= 0.1f * b;
            sgd.step();
            assertEquals(expected, p.value().getArrayUnsafe()[0], "after step " + (step + 1));
        }
    }

    @Test
    void adamMatchesTheTextbookFormula() {
        // the implementation pulls both bias corrections out of the loop; this checks
        // that the regrouping is the same expression, against a double-precision
        // mHat / (sqrt(vHat) + eps) written out step by step
        int n = 6;
        float rate = 0.01f;
        MatrixF start = Matrices.randomUniformF(n, 1, -1.0f, 1.0f, 95L);
        Parameter p = new Parameter("w", start.copy(), true);
        Adam adam = new Adam(rate);
        adam.add(p);

        double[] reference = new double[n];
        for (int i = 0; i < n; ++i) {
            reference[i] = start.getUnsafe(i, 0);
        }
        double[] m = new double[n];
        double[] v = new double[n];

        for (int t = 1; t <= 5; ++t) {
            MatrixF g = Matrices.randomUniformF(n, 1, -0.5f, 0.5f, 96L + t);
            p.grad().setInplace(g);
            adam.step();
            for (int i = 0; i < n; ++i) {
                double grad = g.getUnsafe(i, 0);
                m[i] = 0.9 * m[i] + 0.1 * grad;
                v[i] = 0.999 * v[i] + 0.001 * grad * grad;
                double mHat = m[i] / (1.0 - Math.pow(0.9, t));
                double vHat = v[i] / (1.0 - Math.pow(0.999, t));
                reference[i] -= rate * mHat / (Math.sqrt(vHat) + 1e-8);
            }
        }
        for (int i = 0; i < n; ++i) {
            assertEquals(reference[i], p.value().getUnsafe(i, 0), 1e-5, "element " + i);
        }
    }

    @Test
    void adamWithDecayShrinksAParameterThatHasNoGradient() {
        // the signature of decoupled decay: L2 folded into the gradient could not do
        // this, because a zero gradient would stay zero after the fold
        Parameter decayed = new Parameter("w", Matrices.createF(1, 1), true);
        Parameter exempt = new Parameter("b", Matrices.createF(1, 1), false);
        decayed.value().getArrayUnsafe()[0] = 2.0f;
        exempt.value().getArrayUnsafe()[0] = 2.0f;

        Adam adam = new Adam(0.1f, 0.5f);
        adam.add(decayed);
        adam.add(exempt);
        adam.step();

        assertEquals(2.0f * (1.0f - 0.1f * 0.5f), decayed.value().getArrayUnsafe()[0]);
        assertEquals(2.0f, exempt.value().getArrayUnsafe()[0], "biases and scales are never decayed");
    }

    @Test
    void adamWithoutDecayLeavesAParameterThatHasNoGradientAlone() {
        Parameter p = new Parameter("w", Matrices.createF(1, 1), true);
        p.value().getArrayUnsafe()[0] = 2.0f;
        Adam adam = new Adam(0.1f);
        adam.add(p);
        adam.step();
        adam.step();

        assertEquals(2.0f, p.value().getArrayUnsafe()[0]);
    }

    @Test
    void adamRejectsAParameterTwiceLikeSgdDoes() {
        Parameter p = new Parameter("w", Matrices.createF(2, 2), true);
        Adam adam = new Adam(0.01f);
        adam.add(p);
        assertThrows(IllegalArgumentException.class, () -> adam.add(p));
    }

    @Test
    void theGradientNormSpansEveryRegisteredParameter() {
        // 3-4-5 and 5-12-13: every intermediate value is exact in float, so a wrong norm
        // cannot hide behind rounding. A single-parameter version of this test would
        // pass even if the loop visited only the first parameter.
        Parameter first = new Parameter("w", Matrices.createF(2, 2), true);
        first.grad().getArrayUnsafe()[0] = 3.0f;
        first.grad().getArrayUnsafe()[3] = -4.0f;
        Parameter second = new Parameter("b", Matrices.createF(1, 1), false);
        second.grad().getArrayUnsafe()[0] = 12.0f;
        Sgd sgd = new Sgd(RATE);
        sgd.add(first);

        assertEquals(5.0f, sgd.gradientNorm());
        sgd.add(second);
        assertEquals(13.0f, sgd.gradientNorm());
    }

    @Test
    void theGradientNormOfAnOptimizerWithoutParametersIsZero() {
        assertEquals(0.0f, new Sgd(RATE).gradientNorm());
    }

    @Test
    void clippingIsOffByDefaultAndAHugeGradientPassesThrough() {
        Parameter p = new Parameter("w", Matrices.createF(1, 1), true);
        p.grad().getArrayUnsafe()[0] = 1000.0f;
        Sgd sgd = new Sgd(1.0f);
        sgd.add(p);
        sgd.step();

        assertEquals(-1000.0f, p.value().getArrayUnsafe()[0]);
        assertEquals(0, sgd.clippedSteps());
    }

    @Test
    void aGradientBelowTheThresholdIsNotTouchedAtAll() {
        Parameter p = new Parameter("w", Matrices.createF(1, 2), true);
        float[] g = p.grad().getArrayUnsafe();
        g[0] = -0.0f;
        g[1] = 1e-3f;
        int[] before = bits(g);

        Sgd sgd = new Sgd(0.0f);
        sgd.clipGradientNorm(1.0f);
        sgd.add(p);
        sgd.step();

        for (int i = 0; i < before.length; ++i) {
            assertEquals(before[i], Float.floatToRawIntBits(g[i]), "element " + i);
        }
        assertEquals(0, sgd.clippedSteps());
    }

    @Test
    void aGradientAboveTheThresholdIsScaledToExactlyTheThreshold() {
        // 6-8-10 clipped to 5 is 3-4-5, and every value in it is exact in float
        Parameter p = new Parameter("w", Matrices.createF(1, 2), true);
        p.grad().getArrayUnsafe()[0] = 6.0f;
        p.grad().getArrayUnsafe()[1] = 8.0f;

        Sgd sgd = new Sgd(0.0f);
        sgd.clipGradientNorm(5.0f);
        sgd.add(p);
        sgd.step();

        assertEquals(3.0f, p.grad().getArrayUnsafe()[0]);
        assertEquals(4.0f, p.grad().getArrayUnsafe()[1]);
        assertEquals(5.0f, sgd.gradientNorm());
        assertEquals(1, sgd.clippedSteps());
    }

    @Test
    void clippingScalesEveryParameterByTheSameFactor() {
        // one norm over both, not one norm each: clipping them separately would leave
        // the first untouched and scale only the second
        Parameter first = new Parameter("w", Matrices.createF(1, 2), true);
        first.grad().getArrayUnsafe()[0] = 3.0f;
        first.grad().getArrayUnsafe()[1] = 4.0f;
        Parameter second = new Parameter("b", Matrices.createF(1, 1), false);
        second.grad().getArrayUnsafe()[0] = 12.0f;

        Sgd sgd = new Sgd(0.0f);
        sgd.clipGradientNorm(6.5f);
        sgd.add(first);
        sgd.add(second);
        sgd.step();

        assertEquals(1.5f, first.grad().getArrayUnsafe()[0]);
        assertEquals(2.0f, first.grad().getArrayUnsafe()[1]);
        assertEquals(6.0f, second.grad().getArrayUnsafe()[0]);
    }

    @Test
    void momentumAccumulatesTheClippedGradient() {
        // red if the factor were handed to update() instead of applied to the buffers:
        // the velocity would then accumulate the unclipped gradient
        Parameter p = new Parameter("w", Matrices.createF(1, 1), true);
        Sgd sgd = new Sgd(0.5f, 0.5f);
        sgd.clipGradientNorm(2.0f);
        sgd.add(p);

        float b = 0.0f;
        float expected = 0.0f;
        for (int step = 0; step < 3; ++step) {
            p.grad().getArrayUnsafe()[0] = 10.0f;
            b = 0.5f * b + 2.0f;
            expected -= 0.5f * b;
            sgd.step();
            assertEquals(expected, p.value().getArrayUnsafe()[0], "after step " + (step + 1));
        }
        assertEquals(3, sgd.clippedSteps());
    }

    @Test
    void adamDecaysAParameterAtFullStrengthEvenWhenItsGradientIsClipped() {
        // decoupled decay is computed from the rate alone, so clipping must not reach
        // it: on a hard-clipped step the Adam term shrinks and the decay does not
        assertEquals(decayedValueAfterOneAdamStep(Float.POSITIVE_INFINITY),
                decayedValueAfterOneAdamStep(1e-3f));
    }

    private static float decayedValueAfterOneAdamStep(float maxNorm) {
        Parameter loud = new Parameter("w", Matrices.createF(1, 1), true);
        loud.grad().getArrayUnsafe()[0] = 1000.0f;
        Parameter quiet = new Parameter("q", Matrices.createF(1, 1), true);
        quiet.value().getArrayUnsafe()[0] = 2.0f;
        Adam adam = new Adam(0.1f, 0.5f);
        if (maxNorm != Float.POSITIVE_INFINITY) {
            adam.clipGradientNorm(maxNorm);
        }
        adam.add(loud);
        adam.add(quiet);
        adam.step();
        return quiet.value().getArrayUnsafe()[0];
    }

    @Test
    void aNonFiniteGradientIsNotClipped() {
        // the gradients pass through untouched so that a divergence stays visible
        // instead of being scaled away
        assertNotClipped(Float.NaN);
        assertNotClipped(Float.POSITIVE_INFINITY);
    }

    private static void assertNotClipped(float poison) {
        Parameter p = new Parameter("w", Matrices.createF(1, 2), true);
        float[] g = p.grad().getArrayUnsafe();
        g[0] = poison;
        g[1] = 1.0f;
        int[] before = bits(g);

        Sgd sgd = new Sgd(0.0f);
        sgd.clipGradientNorm(1.0f);
        sgd.add(p);
        sgd.step();

        for (int i = 0; i < before.length; ++i) {
            assertEquals(before[i], Float.floatToRawIntBits(g[i]), "element " + i);
        }
        assertEquals(0, sgd.clippedSteps());
    }

    @Test
    void aNormAboveTheFloatRangeIsStillClipped() {
        // the true norm is 4.8e38, which is +Infinity as a float; narrowing it would let
        // the one case clipping exists for be the one case that escapes
        Parameter p = new Parameter("w", Matrices.createF(1, 2), true);
        p.grad().getArrayUnsafe()[0] = Float.MAX_VALUE;
        p.grad().getArrayUnsafe()[1] = Float.MAX_VALUE;

        Sgd sgd = new Sgd(0.0f);
        sgd.clipGradientNorm(1.0f);
        sgd.add(p);
        sgd.step();

        assertEquals(1, sgd.clippedSteps());
        assertEquals(1.0f, sgd.gradientNorm(), 1e-6f);
        assertEquals((float) (1.0 / Math.sqrt(2.0)), p.grad().getArrayUnsafe()[0], 1e-6f);
    }

    @Test
    void aRateOfZeroStillClipsTheGradientButMovesNoParameter() {
        Parameter p = new Parameter("w", Matrices.createF(1, 2), true);
        p.grad().getArrayUnsafe()[0] = 6.0f;
        p.grad().getArrayUnsafe()[1] = 8.0f;

        Sgd sgd = new Sgd(0.0f);
        sgd.clipGradientNorm(5.0f);
        sgd.add(p);
        sgd.step();

        assertEquals(3.0f, p.grad().getArrayUnsafe()[0]);
        assertEquals(0.0f, p.value().getArrayUnsafe()[0]);
        assertEquals(0.0f, p.value().getArrayUnsafe()[1]);
    }

    @Test
    void theLastGradientNormIsTheValueBeforeClipping() {
        Parameter p = new Parameter("w", Matrices.createF(1, 2), true);
        p.grad().getArrayUnsafe()[0] = 6.0f;
        p.grad().getArrayUnsafe()[1] = 8.0f;

        Sgd sgd = new Sgd(0.0f);
        sgd.clipGradientNorm(5.0f);
        sgd.add(p);
        sgd.step();

        assertEquals(10.0f, sgd.lastGradientNorm());
        assertEquals(5.0f, sgd.gradientNorm());
    }

    @Test
    void clippedStepsCountsOnlyTheStepsThatFired() {
        Parameter p = new Parameter("w", Matrices.createF(1, 1), true);
        Sgd sgd = new Sgd(0.0f);
        sgd.clipGradientNorm(1.0f);
        sgd.add(p);

        p.grad().getArrayUnsafe()[0] = 0.5f;
        sgd.step();
        assertEquals(0, sgd.clippedSteps());

        p.grad().getArrayUnsafe()[0] = 4.0f;
        sgd.step();
        assertEquals(1, sgd.clippedSteps());

        p.grad().getArrayUnsafe()[0] = 1.0f;
        sgd.step();
        assertEquals(1, sgd.clippedSteps(), "a norm exactly at the bound is not clipped");
        assertEquals(3, sgd.steps());
    }

    @Test
    void aClipThresholdThatIsNotPositiveIsRejected() {
        // a negative threshold would make the factor negative and every step ascent
        Sgd sgd = new Sgd(RATE);
        assertThrows(IllegalArgumentException.class, () -> sgd.clipGradientNorm(-1.0f));
        assertThrows(IllegalArgumentException.class, () -> sgd.clipGradientNorm(0.0f));
        assertThrows(IllegalArgumentException.class, () -> sgd.clipGradientNorm(-0.0f));
        assertThrows(IllegalArgumentException.class, () -> sgd.clipGradientNorm(Float.NaN));
    }

    private static int[] bits(float[] values) {
        int[] raw = new int[values.length];
        for (int i = 0; i < values.length; ++i) {
            raw[i] = Float.floatToRawIntBits(values[i]);
        }
        return raw;
    }

    private static void assertBitwise(MatrixF expected, MatrixF actual) {
        float[] e = expected.getArrayUnsafe();
        float[] a = actual.getArrayUnsafe();
        assertEquals(e.length, a.length);
        for (int i = 0; i < e.length; ++i) {
            assertEquals(e[i], a[i], "element " + i);
        }
    }
}
