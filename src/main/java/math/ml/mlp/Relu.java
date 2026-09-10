/*
 * Copyright 2024 Stefan Zobel
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

import math.dl.RELU;

/** Rectified linear unit, {@code max(0, x)}. */
public class Relu extends Activation {

    /** Creates a ReLU layer. */
    public Relu() {
        super(RELU::reluF, RELU::dreluF_dx);
    }

    @Override
    void applyForward(float[] in, float[] out, int from, int to) {
        for (int i = from; i < to; ++i) {
            out[i] = RELU.reluF(in[i]);
        }
    }

    @Override
    void applyBackward(float[] preAct, float[] grads, float[] out, int from, int to) {
        // deliberately not "preAct[i] > 0 ? grads[i] : 0": dreluF_dx propagates NaN,
        // and the product yields -0.0f where a select would yield +0.0f
        for (int i = from; i < to; ++i) {
            out[i] = grads[i] * RELU.dreluF_dx(preAct[i]);
        }
    }
}
