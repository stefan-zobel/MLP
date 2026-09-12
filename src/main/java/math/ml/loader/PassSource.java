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
package math.ml.loader;

import java.util.SplittableRandom;

import net.jamu.matrix.MatrixF;

/**
 * A training set that hands out one pass over its images at a time. Implementations differ in
 * where a pass comes from, which is what lets a network train against either without changing.
 */
public interface PassSource {

    /**
     * The images of the latest pass, one per column, scaled into {@code [0, 1]}.
     *
     * @return the image matrix, refilled by every {@link #regenerate(SplittableRandom)}
     */
    MatrixF images();

    /**
     * The labels belonging to {@link #images()}, column by column.
     *
     * @return the label matrix, refilled by every {@link #regenerate(SplittableRandom)}
     */
    MatrixF labels();

    /**
     * Refills {@link #images()} and {@link #labels()} with the next pass, in which every image
     * appears exactly once.
     *
     * @param rnd source of the order and of whatever else the pass draws
     */
    void regenerate(SplittableRandom rnd);
}
