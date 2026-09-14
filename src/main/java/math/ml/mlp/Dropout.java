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

import java.io.IOException;
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

    /** Names the bundle entry of this layer; without a name it has none. */
    private final String name;

    /** How far {@link #rng} has been drawn, which is the whole of the resumable state. */
    private long drawn;

    /**
     * Creates a Dropout layer with an unseeded mask sequence.
     *
     * @param dropoutRate fraction of the activations to zero out
     */
    public Dropout(float dropoutRate) {
        this(dropoutRate, null, ThreadLocalRandom.current().nextLong());
    }

    /**
     * A fresh mask is drawn per forward pass, so {@code seed} fixes the whole
     * sequence of masks. INFER mode and rate 0 draw nothing.
     *
     * @param dropoutRate fraction of the activations to zero out
     * @param seed        seed for the mask sequence
     */
    public Dropout(float dropoutRate, long seed) {
        this(dropoutRate, null, seed);
    }

    /**
     * A named layer takes its place in the bundle of its network, which is what lets a
     * continued run draw the same masks the interrupted one would have drawn next.
     *
     * @param dropoutRate fraction of the activations to zero out
     * @param name        names the bundle entry of this layer
     * @param seed        seed for the mask sequence
     */
    public Dropout(float dropoutRate, String name, long seed) {
        this.dropoutRate = dropoutRate;
        this.scalingFactor = 1.0f / (1.0f - dropoutRate);
        this.name = name;
        this.rng = new SplittableRandom(seed);
    }

    /**
     * The one place a value leaves {@link #rng}, so that the forward pass and the
     * replay below cannot draw differently.
     */
    private long nextSeed() {
        ++drawn;
        return rng.nextLong();
    }

    @Override
    public void writeParameters(ParameterSink sink) throws IOException {
        ParameterStore.requireName(name, this);
        ParameterStore.writeLong(sink, name + "/draws", drawn);
    }

    @Override
    public void readParameters(ParameterSource source) throws IOException {
        ParameterStore.requireName(name, this);
        long target = ParameterStore.readLong(source, name + "/draws");
        if (target < drawn) {
            throw new IllegalStateException(
                    name + " has drawn " + drawn + " masks and the bundle stops at " + target);
        }
        // a mask sequence only runs forwards: drawing it down is the same as seeding a
        // fresh generator and replaying, and it keeps the generator final
        while (drawn < target) {
            nextSeed();
        }
    }

    /** Answers from the rate, not from the mode, so it is stable before training starts. */
    @Override
    public boolean mutatesInput() {
        return dropoutRate > 0.0f;
    }

    @Override
    public boolean mutatesGradients() {
        return dropoutRate > 0.0f;
    }

    // input: j x m, modified in place
    @Override
    public MatrixF forward(MatrixF input) {
        if (mode == NetworkMode.INFER || dropoutRate <= 0.0f) {
            return input;
        }
        // One pass instead of three. The draw, the threshold and the masking all walk
        // the same j x m elements, and going through an FFunction to threshold is what
        // puts this layer on the shared, megamorphic map call site.
        mask = Matrices.randomUniformF(input.numRows(), input.numColumns(), 0.0f, 1.0f, nextSeed());
        float[] m = mask.getArrayUnsafe();
        float[] x = input.getArrayUnsafe();
        for (int i = 0; i < m.length; ++i) {
            float factor = m[i] < dropoutRate ? 0.0f : scalingFactor;
            m[i] = factor;
            x[i] *= factor;
        }
        return input;
    }

    @Override
    public MatrixF backward(MatrixF grads) {
        if (mode == NetworkMode.INFER) {
            return null;
        }
        if (dropoutRate <= 0.0f) {
            return grads;
        }
        return grads.hadamard(mask, grads);
    }
}