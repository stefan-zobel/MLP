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

import java.util.SplittableRandom;
import java.util.concurrent.ThreadLocalRandom;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

/**
 * Inverted dropout: zeroes a random fraction of the activations during training
 * and scales the survivors so that the expected activation is unchanged.
 */
public class Dropout extends AbstractLayer {

    private final float dropoutRate;
    private final float scalingFactor;

    /**
     * Zero where a unit was dropped, {@code scalingFactor} elsewhere. Held
     * between the passes because backward() has to mask exactly the same units.
     */
    private MatrixF mask;

    /** Owned by this layer, so two Dropout layers never interfere. */
    private final SplittableRandom rng;

    /**
     * Creates a Dropout layer with an unseeded mask sequence.
     *
     * @param dropoutRate fraction of the activations to zero out
     */
    public Dropout(float dropoutRate) {
        this(dropoutRate, ThreadLocalRandom.current().nextLong());
    }

    /**
     * A fresh mask is drawn per forward pass, so {@code seed} fixes the whole
     * sequence of masks. INFER mode and rate 0 draw nothing.
     *
     * @param dropoutRate fraction of the activations to zero out
     * @param seed        seed for the mask sequence
     */
    public Dropout(float dropoutRate, long seed) {
        this.dropoutRate = dropoutRate;
        this.scalingFactor = 1.0f / (1.0f - dropoutRate);
        this.rng = new SplittableRandom(seed);
    }

    // input: j x m, modified in place
    @Override
    public MatrixF forward(MatrixF input) {
        if (mode == NetworkMode.INFER || dropoutRate <= 0.0f) {
            return input;
        }
        float rate = dropoutRate;
        float scale = scalingFactor;
        mask = Matrices.randomUniformF(input.numRows(), input.numColumns(), 0.0f, 1.0f, rng.nextLong())
                .mapInplace(u -> u < rate ? 0.0f : scale);
        return input.hadamard(mask, input);
    }

    @Override
    public MatrixF backward(MatrixF grads, float unused) {
        if (mode == NetworkMode.INFER) {
            return null;
        }
        if (dropoutRate <= 0.0f) {
            return grads;
        }
        return grads.hadamard(mask, grads);
    }
}