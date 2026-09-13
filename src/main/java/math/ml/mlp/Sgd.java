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

/**
 * Stochastic gradient descent, {@code p -= rate * dL/dp}, optionally with a
 * momentum term {@code b = momentum * b + dL/dp} that is used in its place.
 */
public final class Sgd extends AbstractOptimizer {

    /** One velocity buffer per parameter; empty when there is no momentum. */
    private final List<float[]> velocity = new ArrayList<>();

    private final float momentum;
    private float rate;

    /**
     * Plain SGD at a fixed learning rate.
     *
     * @param rate the learning rate ({@code 0 <= r < 1})
     */
    public Sgd(float rate) {
        this(LearningRateSchedule.constant(rate), 0.0f);
    }

    /**
     * SGD with momentum at a fixed learning rate.
     *
     * @param rate     the learning rate ({@code 0 <= r < 1})
     * @param momentum the momentum ({@code 0} for plain SGD), typically {@code 0.9f}
     */
    public Sgd(float rate, float momentum) {
        this(LearningRateSchedule.constant(rate), momentum);
    }

    /**
     * Plain SGD at a rate that varies with the step.
     *
     * @param schedule supplies the learning rate of every step
     */
    public Sgd(LearningRateSchedule schedule) {
        this(schedule, 0.0f);
    }

    /**
     * SGD with momentum at a rate that varies with the step.
     *
     * @param schedule supplies the learning rate of every step
     * @param momentum the momentum ({@code 0} for plain SGD)
     */
    public Sgd(LearningRateSchedule schedule, float momentum) {
        super(schedule);
        this.momentum = momentum;
    }

    @Override
    protected void registered(Parameter p) {
        if (momentum != 0.0f) {
            velocity.add(new float[p.value().getArrayUnsafe().length]);
        }
    }

    @Override
    protected void beginStep(float rate, int step) {
        this.rate = rate;
    }

    @Override
    protected void update(Parameter p, int index) {
        float[] v = p.value().getArrayUnsafe();
        float[] g = p.grad().getArrayUnsafe();

        if (momentum == 0.0f) {
            if (rate == 0.0f) {
                // MatrixF.addInplace skips its loop at alpha zero, and -0.0f == 0.0f, so
                // both zeros are a no-op there. Writing anyway would turn a -0.0f
                // parameter into +0.0f and a non-finite gradient into a non-finite
                // parameter.
                return;
            }
            for (int i = 0; i < v.length; ++i) {
                v[i] -= rate * g[i];
            }
            return;
        }

        // with momentum the buffer keeps accumulating even at a rate of zero, so that a
        // warmup starting at zero does not silently drop the first gradients
        float[] b = velocity.get(index);
        boolean move = rate != 0.0f;
        for (int i = 0; i < v.length; ++i) {
            b[i] = momentum * b[i] + g[i];
            if (move) {
                v[i] -= rate * b[i];
            }
        }
    }
}
