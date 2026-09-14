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
import java.io.OutputStream;

/**
 * Where a layer writes its persistent state, one named entry per matrix.
 */
public interface ParameterSink {

    /**
     * Opens the entry {@code key} for writing. The caller closes the stream,
     * which ends the entry without closing whatever holds it.
     *
     * @param key the entry name, by convention the layer name, a slash and the
     *            name of the parameter
     * @return the stream to serialize into
     * @throws IOException          if the entry cannot be opened
     * @throws IllegalStateException if {@code key} was written before
     */
    OutputStream open(String key) throws IOException;
}
