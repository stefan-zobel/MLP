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
 * Weight initialization schemes for {@link Hidden}. Both draw uniformly from
 * {@code [-bound, bound]} and differ only in that bound.
 */
public enum Init {

    /**
     * Glorot (Xavier) uniform: {@code sqrt(6 / (fan_in + fan_out))}. Assumes an
     * activation symmetric about zero, which fits a linear output or a sigmoid.
     */
    GLOROT {
        @Override
        float bound(int fanIn, int fanOut) {
            return (float) Math.sqrt(6.0 / (fanIn + fanOut));
        }
    },

    /**
     * He (Kaiming) uniform: {@code sqrt(6 / fan_in)}. The right choice ahead of
     * ReLU or GELU, which discard the negative half of the pre-activations.
     */
    HE {
        @Override
        float bound(int fanIn, int fanOut) {
            return (float) Math.sqrt(6.0 / fanIn);
        }
    };

    abstract float bound(int fanIn, int fanOut);
}
