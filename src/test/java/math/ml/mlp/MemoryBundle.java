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

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

// a bundle that never reaches the filesystem. Two views rather than one object, because the
// two open() methods differ only in their return type and Java will not have both.
final class MemoryBundle {

    private final Map<String, byte[]> entries = new LinkedHashMap<>();

    ParameterSink sink() {
        return key -> {
            if (entries.containsKey(key)) {
                throw new IllegalStateException(key + " was written before; two layers cannot share a name");
            }
            return new ByteArrayOutputStream() {
                @Override
                public void close() {
                    entries.put(key, toByteArray());
                }
            };
        };
    }

    ParameterSource source() {
        return key -> {
            byte[] bytes = entries.get(key);
            if (bytes == null) {
                throw new IllegalStateException("this bundle holds no entry " + key);
            }
            return new ByteArrayInputStream(bytes);
        };
    }

    Set<String> keys() {
        return entries.keySet();
    }

    int size() {
        return entries.size();
    }

    byte[] bytes(String key) {
        return entries.get(key);
    }

    void put(String key, byte[] value) {
        entries.put(key, value);
    }

    void remove(String key) {
        entries.remove(key);
    }
}
