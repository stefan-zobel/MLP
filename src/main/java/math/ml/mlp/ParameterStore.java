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

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

/** Where a model bundle is read from and written to, and the two calls a layer makes. */
final class ParameterStore {

    /**
     * Curated models, and the data sets of the loaders. No training run writes a
     * bundle here, so one promoted by hand stays what it was.
     */
    static final String LOAD_DIR = "./data/";

    /**
     * Where training runs write. Promote a checkpoint to {@link #LOAD_DIR} manually
     * once it has proven itself, which is now one file.
     */
    static final String STORE_DIR = "./checkpoints/";

    /**
     * Serializes one matrix into the entry {@code key}.
     *
     * @param sink   where the entry goes
     * @param key    the entry name
     * @param matrix the matrix to write
     * @throws IOException if writing fails
     */
    static void write(ParameterSink sink, String key, MatrixF matrix) throws IOException {
        try (OutputStream os = new BufferedOutputStream(sink.open(key))) {
            Matrices.serializeF(matrix, os);
        }
    }

    /**
     * Reads the entry {@code key} into a matrix that already has the right shape,
     * so that a bundle from a differently sized network is refused here.
     *
     * @param source where the entry comes from
     * @param key    the entry name
     * @param into   the matrix to fill
     * @throws IOException if reading fails
     */
    static void read(ParameterSource source, String key, MatrixF into) throws IOException {
        try (InputStream is = new BufferedInputStream(source.open(key))) {
            into.setInplace(Matrices.deserializeF(is));
        }
    }

    /**
     * Refuses a layer that is asked to take part in a bundle without a name. Being
     * nameless is legal until that moment; contributing nothing to a bundle is not,
     * because a model missing some of its weights looks exactly like one that is not.
     *
     * @param name  the name of the layer, or null
     * @param layer the layer, for the message
     * @throws IllegalStateException if {@code name} is null
     */
    static void requireName(String name, Layer layer) {
        if (name == null) {
            throw new IllegalStateException(layer.getClass().getSimpleName()
                    + " has trainable parameters but no name, so it cannot be part of a bundle");
        }
    }

    private ParameterStore() {
        throw new AssertionError();
    }
}
