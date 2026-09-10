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
}
