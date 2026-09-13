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
 * Adam (Kingma &amp; Ba, 2015) with <em>decoupled</em> weight decay, so that a
 * nonzero decay makes this AdamW (Loshchilov &amp; Hutter, 2019) rather than L2
 * regularization folded into the gradient. Parameters that answer {@code false} to
 * {@link Parameter#isDecayed()} are never decayed.
 */
public final class Adam extends AbstractOptimizer {

    /** The two moment estimates, one array per registered parameter. */
    private final List<float[]> firstMoment = new ArrayList<>();
    private final List<float[]> secondMoment = new ArrayList<>();

    private final float beta1;
    private final float beta2;
    private final float eps;
    private final float weightDecay;

    // recomputed once per step by beginStep()
    private float alpha;
    private float epsHat;
    private float decayFactor;

    /**
     * Adam with the usual defaults {@code beta1 = 0.9}, {@code beta2 = 0.999},
     * {@code eps = 1e-8} and no weight decay.
     *
     * @param rate the learning rate
     */
    public Adam(float rate) {
        this(LearningRateSchedule.constant(rate), 0.0f);
    }

    /**
     * Adam with decoupled weight decay, that is AdamW.
     *
     * @param rate        the learning rate
     * @param weightDecay decay applied to the decayed parameters, typically {@code 0.01f}
     */
    public Adam(float rate, float weightDecay) {
        this(LearningRateSchedule.constant(rate), weightDecay);
    }

    /**
     * Adam with a rate that varies with the step.
     *
     * @param schedule    supplies the learning rate of every step
     * @param weightDecay decay applied to the decayed parameters, {@code 0} for plain Adam
     */
    public Adam(LearningRateSchedule schedule, float weightDecay) {
        this(schedule, weightDecay, 0.9f, 0.999f, 1e-8f);
    }

    /**
     * Adam with every constant spelled out.
     *
     * @param schedule    supplies the learning rate of every step
     * @param weightDecay decay applied to the decayed parameters, {@code 0} for plain Adam
     * @param beta1       decay of the first moment, usually {@code 0.9f}
     * @param beta2       decay of the second moment, usually {@code 0.999f}
     * @param eps         added to the square root of the second moment
     */
    public Adam(LearningRateSchedule schedule, float weightDecay, float beta1, float beta2, float eps) {
        super(schedule);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weightDecay = weightDecay;
    }

    @Override
    protected void registered(Parameter p) {
        int size = p.value().getArrayUnsafe().length;
        firstMoment.add(new float[size]);
        secondMoment.add(new float[size]);
    }

    @Override
    protected void beginStep(float rate, int step) {
        // The textbook form is p -= rate * mHat / (sqrt(vHat) + eps) with
        // mHat = m/bc1 and vHat = v/bc2. Pulling both corrections out of the loop
        // turns that into alpha * m / (sqrt(v) + epsHat) with the two constants below,
        // which is the same expression and one division per element instead of three.
        float bc1 = 1.0f - (float) Math.pow(beta1, step);
        float rootBc2 = (float) Math.sqrt(1.0f - (float) Math.pow(beta2, step));
        alpha = rate * rootBc2 / bc1;
        epsHat = eps * rootBc2;
        decayFactor = 1.0f - rate * weightDecay;
    }

    @Override
    protected void update(Parameter p, int index) {
        float[] v = p.value().getArrayUnsafe();
        float[] g = p.grad().getArrayUnsafe();
        float[] m1 = firstMoment.get(index);
        float[] m2 = secondMoment.get(index);
        boolean decay = weightDecay != 0.0f && p.isDecayed();

        for (int i = 0; i < v.length; ++i) {
            float grad = g[i];
            m1[i] = beta1 * m1[i] + (1.0f - beta1) * grad;
            m2[i] = beta2 * m2[i] + (1.0f - beta2) * grad * grad;
            if (decay) {
                // decoupled: the decay never enters the moments, which is the whole
                // point of AdamW and what makes it act even on a zero gradient
                v[i] *= decayFactor;
            }
            v[i] -= alpha * m1[i] / ((float) Math.sqrt(m2[i]) + epsHat);
        }
    }
}
