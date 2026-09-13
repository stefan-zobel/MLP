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

/** The learning rate an optimizer uses for a given training step. */
@FunctionalInterface
public interface LearningRateSchedule {

    /**
     * The rate for one step.
     *
     * @param step the training step, counted from one
     * @return the learning rate to apply
     */
    float rate(int step);

    /**
     * A rate that never changes.
     *
     * @param rate the learning rate for every step
     * @return the schedule
     */
    static LearningRateSchedule constant(float rate) {
        return step -> rate;
    }

    /**
     * A rate rising linearly to {@code peak} over the first {@code warmupSteps} steps
     * and staying there afterwards.
     *
     * @param warmupSteps the number of steps the rise takes, at least one
     * @param peak        the rate reached at the last warmup step
     * @return the schedule
     * @throws IllegalArgumentException if {@code warmupSteps} is smaller than one
     */
    static LearningRateSchedule linearWarmup(int warmupSteps, float peak) {
        if (warmupSteps < 1) {
            throw new IllegalArgumentException("warmupSteps must be at least 1: " + warmupSteps);
        }
        return step -> {
            int s = Math.max(1, step);
            // the cast placement is not cosmetic: peak * ((float) w / w) is exactly peak,
            // whereas (peak * w) / w need not be, and warmupThenCosine hands over there
            return s <= warmupSteps ? peak * ((float) s / warmupSteps) : peak;
        };
    }

    /**
     * A rate following a half cosine from {@code peak} at step one down to
     * {@code minRate} at step {@code totalSteps}, staying there afterwards.
     *
     * @param peak       the rate of the first step
     * @param totalSteps the step at which the rate reaches {@code minRate}, at least two
     * @param minRate    the rate of the last step and of everything past it
     * @return the schedule
     * @throws IllegalArgumentException if {@code totalSteps} is smaller than two or
     *                                  {@code minRate} exceeds {@code peak}
     */
    static LearningRateSchedule cosineDecay(float peak, int totalSteps, float minRate) {
        if (totalSteps < 2) {
            throw new IllegalArgumentException("totalSteps must be at least 2: " + totalSteps);
        }
        if (minRate > peak) {
            throw new IllegalArgumentException("minRate must not exceed peak: " + minRate + " > " + peak);
        }
        return step -> {
            int s = Math.max(1, step);
            if (s >= totalSteps) {
                return minRate;
            }
            // in double, and not because the rate needs it: near the end (1 + cos) cancels
            // catastrophically, and a float computation would leave a test nothing to
            // compare against but the same expression
            double progress = (s - 1) / (double) (totalSteps - 1);
            return (float) (minRate + (peak - minRate) * 0.5 * (1.0 + Math.cos(Math.PI * progress)));
        };
    }

    /**
     * A linear warmup to {@code peak} followed by a cosine decay to {@code minRate},
     * the schedule a transformer is usually trained with.
     *
     * @param warmupSteps the number of steps the rise takes, at least one
     * @param peak        the rate reached at the last warmup step
     * @param totalSteps  the length of the <em>whole</em> run, warmup included
     * @param minRate     the rate of the last step and of everything past it
     * @return the schedule
     * @throws IllegalArgumentException if the two phases do not fit into
     *                                  {@code totalSteps} or {@code minRate} exceeds
     *                                  {@code peak}
     */
    static LearningRateSchedule warmupThenCosine(int warmupSteps, float peak, int totalSteps, float minRate) {
        if (totalSteps - warmupSteps < 2) {
            throw new IllegalArgumentException(
                    "totalSteps must leave at least 2 steps for the decay: " + totalSteps + " - " + warmupSteps);
        }
        LearningRateSchedule warmup = linearWarmup(warmupSteps, peak);
        // totalSteps is the whole run, so the decay gets only what the warmup leaves.
        // Getting that subtraction wrong is silent, which is why this composition is
        // written once here rather than being left to the caller.
        LearningRateSchedule decay = cosineDecay(peak, totalSteps - warmupSteps, minRate);
        return step -> {
            int s = Math.max(1, step);
            return s <= warmupSteps ? warmup.rate(s) : decay.rate(s - warmupSteps);
        };
    }
}
