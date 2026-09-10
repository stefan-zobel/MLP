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

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** Holds the parameter list, the step counter and the schedule of an optimizer. */
public abstract class AbstractOptimizer implements Optimizer {

    /** The registered parameters, in registration order. */
    protected final List<Parameter> parameters = new ArrayList<>();

    private final LearningRateSchedule schedule;
    private int step;

    /**
     * For subclasses.
     *
     * @param schedule supplies the learning rate of every step
     */
    protected AbstractOptimizer(LearningRateSchedule schedule) {
        this.schedule = Objects.requireNonNull(schedule, "schedule");
    }

    @Override
    public final void add(Parameter p) {
        Objects.requireNonNull(p, "parameter");
        for (Parameter known : parameters) {
            // by identity: two parameters of the same shape and name are still two
            if (known == p) {
                throw new IllegalArgumentException("parameter is already registered: " + p);
            }
        }
        parameters.add(p);
        registered(p);
    }

    @Override
    public final void step() {
        float rate = schedule.rate(++step);
        beginStep(rate, step);
        for (int i = 0; i < parameters.size(); ++i) {
            update(parameters.get(i), i);
        }
    }

    /**
     * Called once per parameter, when it is registered, for subclasses that keep
     * per-parameter state. The default does nothing.
     *
     * @param p the parameter that was just added
     */
    protected void registered(Parameter p) {
        // stateless by default
    }

    /**
     * Called once per step, before any parameter is updated.
     *
     * @param rate the learning rate the schedule returned
     * @param step the step number, counted from one
     */
    protected abstract void beginStep(float rate, int step);

    /**
     * Applies the update prepared by the preceding {@link #beginStep} to one parameter.
     *
     * @param p     the parameter to update
     * @param index its position in {@link #parameters}, for per-parameter state
     */
    protected abstract void update(Parameter p, int index);

    /**
     * How many steps have been applied.
     *
     * @return the step counter
     */
    public final int steps() {
        return step;
    }
}
