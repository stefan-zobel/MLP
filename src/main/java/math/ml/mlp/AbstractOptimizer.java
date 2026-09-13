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

    // infinity rather than zero is "off": zero is a threshold someone may legitimately
    // ask for, so it must not double as the sentinel
    private double maxGradNorm = Double.POSITIVE_INFINITY;
    private double lastNorm;
    private int clippedSteps;

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
        // before anything else, so that update() always sees a gradient that already
        // satisfies the bound and both hot loops can stay as they are
        clip();
        float rate = schedule.rate(++step);
        beginStep(rate, step);
        for (int i = 0; i < parameters.size(); ++i) {
            update(parameters.get(i), i);
        }
    }

    /**
     * Scales all gradients down together whenever their global norm exceeds
     * {@code maxNorm}, from now on. Note that this bounds the gradient, not the step:
     * under momentum the velocity still reaches {@code 1 / (1 - momentum)} times the
     * clipped gradient, and the decoupled weight decay of {@link Adam} is unaffected
     * by design.
     *
     * @param maxNorm the largest global gradient norm to leave alone, positive;
     *                {@link Float#POSITIVE_INFINITY} switches clipping off again
     * @return this optimizer
     * @throws IllegalArgumentException if {@code maxNorm} is not positive
     */
    public final AbstractOptimizer clipGradientNorm(float maxNorm) {
        // a negative threshold would make the factor negative and turn every step into
        // gradient ascent, silently; NaN would disable clipping while still paying for
        // the norm; zero would erase every gradient including the non-finite ones this
        // deliberately lets through
        if (!(maxNorm > 0.0f)) {
            throw new IllegalArgumentException("maxNorm must be positive: " + maxNorm);
        }
        this.maxGradNorm = maxNorm;
        return this;
    }

    private void clip() {
        if (maxGradNorm == Double.POSITIVE_INFINITY) {
            return;
        }
        double norm = globalGradientNorm();
        lastNorm = norm;
        // A non-finite norm is left alone: every term is non-negative, so the sum is
        // infinite or NaN only because an element is, and scaling those away would hide
        // a divergence rather than report it. NaN also fails the comparison by itself.
        if (!(norm > maxGradNorm) || Double.isInfinite(norm)) {
            return;
        }
        ++clippedSteps;
        // the factor stays double: a finite norm can reach 1e43, where the float
        // reciprocal would be subnormal and lose most of its significand
        double factor = maxGradNorm / norm;
        for (int i = 0; i < parameters.size(); ++i) {
            float[] g = parameters.get(i).grad().getArrayUnsafe();
            for (int k = 0; k < g.length; ++k) {
                g[k] = (float) (g[k] * factor);
            }
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
     * The Euclidean norm of the gradients of every registered parameter, taken
     * together as one vector.
     *
     * @return the global gradient norm
     */
    public final float gradientNorm() {
        return (float) globalGradientNorm();
    }

    /**
     * The global gradient norm of the last step, as it was <em>before</em> clipping.
     *
     * @return the norm, or zero while clipping is switched off
     */
    public final float lastGradientNorm() {
        return (float) lastNorm;
    }

    /**
     * How many steps were scaled down because their gradients exceeded the bound.
     *
     * @return the number of clipped steps
     */
    public final int clippedSteps() {
        return clippedSteps;
    }

    // In double, and not for overflow: the square of the largest float is about 1.2e77,
    // so millions of them still sum comfortably. MatrixF.normF() would do the
    // overflow-resistant scaled algorithm instead, at a branch and two divides per
    // element, and it is per matrix while this norm spans all of them.
    private double globalGradientNorm() {
        double sum = 0.0;
        for (int i = 0; i < parameters.size(); ++i) {
            float[] g = parameters.get(i).grad().getArrayUnsafe();
            for (int k = 0; k < g.length; ++k) {
                sum += (double) g[k] * g[k];
            }
        }
        return Math.sqrt(sum);
    }

    /**
     * How many steps have been applied.
     *
     * @return the step counter
     */
    public final int steps() {
        return step;
    }
}
