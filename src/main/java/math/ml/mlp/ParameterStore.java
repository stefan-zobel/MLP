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

/** The two directories every layer and the optimizer read from and write to. */
final class ParameterStore {

    /**
     * Curated parameters. Read-only from code, so that no training run can
     * overwrite a set that was promoted here by hand.
     */
    static final String LOAD_DIR = "./data/";

    /**
     * Where training runs write. Promote a checkpoint to {@link #LOAD_DIR}
     * manually once it has proven itself.
     */
    static final String STORE_DIR = "./checkpoints/";

    private ParameterStore() {
        throw new AssertionError();
    }
}
