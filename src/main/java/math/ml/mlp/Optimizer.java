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

import java.io.IOException;

/**
 * Applies the gradients a backward pass produced to the parameters that produced
 * them. The network registers every parameter once and calls {@link #step()} after
 * each backward pass.
 */
public interface Optimizer {

    /**
     * Registers a parameter to be updated from now on.
     *
     * @param p the parameter
     * @throws IllegalArgumentException if this very parameter is already registered
     */
    void add(Parameter p);

    /** Applies one update to every registered parameter. */
    void step();

    /**
     * Writes whatever this optimizer would need to continue into {@code sink}.
     *
     * @param sink where the entry goes
     * @throws IOException if writing fails
     */
    default void writeTo(ParameterSink sink) throws IOException {
        // no-op by default
    }

    /**
     * Reads back what {@link #writeTo(ParameterSink)} wrote and continues at
     * {@code step}, which the bundle holds once for the whole network.
     *
     * @param source where the entry comes from
     * @param step   the step every parameter in the bundle comes from
     * @throws IOException if reading fails
     */
    default void readFrom(ParameterSource source, int step) throws IOException {
        // no-op by default
    }
}
