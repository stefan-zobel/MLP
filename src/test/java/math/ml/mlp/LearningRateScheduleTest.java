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
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import net.jamu.matrix.Matrices;

class LearningRateScheduleTest {

    @Test
    void constantIgnoresTheStep() {
        LearningRateSchedule schedule = LearningRateSchedule.constant(0.05f);

        assertEquals(0.05f, schedule.rate(1));
        assertEquals(0.05f, schedule.rate(1000));
    }

    @Test
    void theFirstWarmupStepIsNeverZero() {
        // written as peak * step / warmupSteps with an integer division this is 0, and a
        // rate of zero is a deliberate no-op in Sgd, so the first batches would be lost
        assertEquals(0.125f, LearningRateSchedule.linearWarmup(4, 0.5f).rate(1));
        assertTrue(LearningRateSchedule.linearWarmup(4000, 3e-4f).rate(1) > 0.0f);
    }

    @Test
    void theWarmupRisesLinearly() {
        LearningRateSchedule schedule = LearningRateSchedule.linearWarmup(4, 0.5f);

        assertEquals(0.125f, schedule.rate(1));
        assertEquals(0.25f, schedule.rate(2));
        assertEquals(0.375f, schedule.rate(3));
    }

    @Test
    void theWarmupReachesThePeakExactlyAtItsLastStep() {
        // (peak * warmupSteps) / warmupSteps is not peak for about one pair in nine, and
        // this is such a pair (that spelling gives 0.46795836f). The handover in
        // warmupThenCosine happens at exactly this step, so it has to be bit-exact.
        assertEquals(0.46795833f, LearningRateSchedule.linearWarmup(6970, 0.46795833f).rate(6970));
        assertEquals(0.5f, LearningRateSchedule.linearWarmup(4, 0.5f).rate(4));
    }

    @Test
    void theWarmupHoldsThePeakAfterItsLastStep() {
        LearningRateSchedule schedule = LearningRateSchedule.linearWarmup(4, 0.5f);

        assertEquals(0.5f, schedule.rate(5));
        assertEquals(0.5f, schedule.rate(100_000));
    }

    @Test
    void aWarmupOfLessThanOneStepIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> LearningRateSchedule.linearWarmup(0, 0.5f));
        assertThrows(IllegalArgumentException.class, () -> LearningRateSchedule.linearWarmup(-1, 0.5f));
    }

    @Test
    void theCosineStartsAtThePeakAndEndsAtTheMinimum() {
        LearningRateSchedule schedule = LearningRateSchedule.cosineDecay(1.0f, 101, 0.25f);

        assertEquals(1.0f, schedule.rate(1));
        assertEquals(0.25f, schedule.rate(101));
    }

    @Test
    void theCosineIsHalfWayDownAtItsMiddleStep() {
        LearningRateSchedule schedule = LearningRateSchedule.cosineDecay(1.0f, 101, 0.0f);

        assertEquals(0.5f, schedule.rate(51), 1e-7f);
    }

    @Test
    void theCosineNeverRisesAndStaysAtTheMinimumPastItsHorizon() {
        LearningRateSchedule schedule = LearningRateSchedule.cosineDecay(1.0f, 101, 0.25f);

        float previous = schedule.rate(1);
        for (int step = 2; step <= 101; ++step) {
            float rate = schedule.rate(step);
            // not strictly decreasing: in the two flat regions the float result moves by
            // an ulp or not at all
            assertTrue(rate <= previous + 1e-7f, "rate rose at step " + step);
            previous = rate;
        }
        assertEquals(0.25f, schedule.rate(5000));
    }

    @Test
    void aCosineOverLessThanTwoStepsIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> LearningRateSchedule.cosineDecay(1.0f, 1, 0.0f));
        assertThrows(IllegalArgumentException.class, () -> LearningRateSchedule.cosineDecay(1.0f, 0, 0.0f));
    }

    @Test
    void aCosineThatWouldRiseIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> LearningRateSchedule.cosineDecay(0.1f, 100, 0.2f));
    }

    @Test
    void warmupThenCosineIsContinuousAtTheHandoverAndEndsAtTheMinimum() {
        // totalSteps is the whole run: the decay must be built over totalSteps -
        // warmupSteps, and handing it the grand total instead ends about 1 % high
        LearningRateSchedule schedule = LearningRateSchedule.warmupThenCosine(10, 0.5f, 110, 0.0f);

        assertEquals(0.5f, schedule.rate(10));
        assertEquals(0.5f, schedule.rate(11));
        assertEquals(0.0f, schedule.rate(110));
        assertEquals(0.0f, schedule.rate(111));
    }

    @Test
    void warmupThenCosineRisesToThePeakBeforeItDecays() {
        LearningRateSchedule schedule = LearningRateSchedule.warmupThenCosine(4, 0.5f, 104, 0.0f);

        assertEquals(0.125f, schedule.rate(1));
        assertEquals(0.375f, schedule.rate(3));
        assertTrue(schedule.rate(12) < 0.5f, "the decay has to start after the warmup");
    }

    @Test
    void warmupThenCosineRejectsATotalThatLeavesNoRoomForTheDecay() {
        assertThrows(IllegalArgumentException.class,
                () -> LearningRateSchedule.warmupThenCosine(10, 0.5f, 11, 0.0f));
        assertThrows(IllegalArgumentException.class,
                () -> LearningRateSchedule.warmupThenCosine(10, 0.5f, 10, 0.0f));
    }

    @Test
    void theScheduleTheOptimizerSeesDrivesTheParameter() {
        LearningRateSchedule schedule = LearningRateSchedule.warmupThenCosine(4, 0.5f, 20, 0.0f);
        Parameter p = new Parameter("w", Matrices.createF(1, 1), true);
        p.grad().getArrayUnsafe()[0] = 1.0f;
        Sgd sgd = new Sgd(schedule);
        sgd.add(p);

        float expected = 0.0f;
        for (int step = 1; step <= 6; ++step) {
            expected -= schedule.rate(step);
            sgd.step();
            assertEquals(expected, p.value().getArrayUnsafe()[0], "after step " + step);
        }
    }
}
