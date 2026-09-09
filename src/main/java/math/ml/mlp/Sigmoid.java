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

/**
 * Sigmoid activation function: &sigma;(x) = 1 / (1 + e<sup>&minus;x</sup>).
 *
 * <p>Maps any real-valued input to the open interval (0, 1), making it the
 * natural output activation for a VAE decoder that reconstructs normalized
 * images (pixel values in [0, 1]).
 *
 * <p>When immediately followed by {@link BinaryCrossEntropyLoss}, the
 * numerically problematic 1/(p&middot;(1&minus;p)) denominator in the BCE gradient cancels
 * with the p&middot;(1&minus;p) factor in the sigmoid derivative, leaving the clean and
 * stable combined gradient <b>p &minus; t</b> at the sigmoid input.
 */
public class Sigmoid extends Activation {

    private static float sigmoid(float x) {
        return 1.0f / (1.0f + (float) Math.exp(-x));
    }

    /**
     * Derivative of sigmoid with respect to its pre-activation input x:
     * &sigma;'(x) = &sigma;(x) &middot; (1 &minus; &sigma;(x)).
     *
     * <p>{@link Activation#backward} applies this function to the cached
     * pre-activation value, so the argument here is x (not &sigma;(x)).
     */
    private static float dsigmoid_dx(float x) {
        float s = sigmoid(x);
        return s * (1.0f - s);
    }

    public Sigmoid() {
        super(Sigmoid::sigmoid, Sigmoid::dsigmoid_dx);
    }
}
