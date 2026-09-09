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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.ListIterator;

import net.jamu.matrix.MatrixF;

/**
 * Residual (skip-connection) layer as introduced by He et al. (2015)
 * &ldquo;Deep Residual Learning for Image Recognition&rdquo;.
 *
 * <p>Wraps an arbitrary sequence of layers {@code F} and adds the identity
 * shortcut to its output:
 * <pre>
 *   y = x + F(x)
 * </pre>
 * The branch transformation {@code F} and the identity path run in parallel;
 * their results are <em>added</em> element-wise (unlike
 * {@link ParallelBranches}, which <em>concatenates</em> vertically).
 *
 * <p><b>Shape constraint:</b> the output of the branch sequence must have the
 * same shape as the input so that the addition is well-defined.  A
 * {@code IllegalStateException} is thrown in {@link #forward} if this
 * constraint is violated.
 *
 * <h2>Forward pass</h2>
 * <pre>
 *   y = x + F(x)
 * </pre>
 * A defensive copy of {@code x} is passed into the branch so that any
 * in-place layer (e.g. {@link Dropout}) cannot corrupt the identity path.
 *
 * <h2>Backward pass</h2>
 * Chain rule for {@code y = x + F(x)}:
 * <pre>
 *   &part;L/&part;x = &part;L/&part;y &middot; &part;y/&part;x = &part;L/&part;y &middot; (I + &part;F/&part;x)
 *          = &part;L/&part;y  +  F.backward(&part;L/&part;y)
 * </pre>
 * The identity path contributes {@code dL/dy} unchanged; the branch
 * contributes the result of its own backward pass through all branch layers.
 * Their sum is returned as the gradient w.r.t. the input.
 *
 * <p>A defensive copy of {@code dL/dy} is also passed to the branch
 * backward so that any in-place backward (e.g. {@link Dropout#backward})
 * cannot corrupt the gradient used for the identity path.
 *
 * <h2>Typical usage</h2>
 * <pre>{@code
 * // Residual block with BN and ReLU inside:
 * net.add(new ResidualBranch(
 *     new Hidden(256, 256, "res1"),
 *     new BatchNorm(256),
 *     new Relu()
 * ));
 * }</pre>
 */
public class ResidualBranch extends AbstractLayer {

    private final List<Layer> branch;

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    /**
     * Constructs a residual branch from a varargs list of layers.
     * The layers are executed in the given order during the forward pass.
     *
     * @param layers the branch transformation F(x)
     */
    public ResidualBranch(Layer... layers) {
        this.branch = new ArrayList<>(Arrays.asList(layers));
    }

    /**
     * Constructs a residual branch from a pre-built list of layers.
     *
     * @param layers the branch transformation F(x)
     */
    public ResidualBranch(List<Layer> layers) {
        this.branch = new ArrayList<>(layers);
    }

    // -------------------------------------------------------------------------
    // Layer contract
    // -------------------------------------------------------------------------

    /**
     * Propagates the network mode to every layer in the branch.
     */
    @Override
    public void setMode(NetworkMode mode) {
        super.setMode(mode); // sets this.mode
        for (Layer layer : branch) {
            layer.setMode(mode);
        }
    }

    /**
     * Delegates parameter persistence to all layers in the branch.
     */
    @Override
    public void storeParameters() {
        for (Layer layer : branch) {
            layer.storeParameters();
        }
    }

    /**
     * Computes {@code y = x + F(x)}.
     *
     * <p>Note: {@code super.forward(input)} is intentionally <em>not</em>
     * called &ndash; {@code ResidualBranch} has no own parameters and the branch
     * layers cache their own inputs.
     *
     * @param input the shared input {@code x}
     * @return {@code x + F(x)}, same shape as {@code input}
     * @throws IllegalStateException if {@code F(x)} has a different shape
     *         than {@code input}
     */
    @Override
    public MatrixF forward(MatrixF input) {
        // Pass a defensive copy into the branch so that in-place ops (e.g.
        // Dropout) cannot corrupt the identity path.
        MatrixF branchOut = copyOf(input);
        for (Layer layer : branch) {
            branchOut = layer.forward(branchOut);
        }

        if (branchOut.numRows() != input.numRows()
                || branchOut.numColumns() != input.numColumns()) {
            throw new IllegalStateException(
                    "ResidualBranch: branch output shape ("
                    + branchOut.numRows() + "x" + branchOut.numColumns()
                    + ") must equal input shape ("
                    + input.numRows() + "x" + input.numColumns() + ")");
        }

        // y = x + F(x)
        return input.plus(branchOut);
    }

    /**
     * Returns {@code dL/dy + F.backward(dL/dy)}.
     *
     * @param grads {@code dL/dy}, the gradient w.r.t. the output of this layer
     * @return gradient w.r.t. the input {@code x}; {@code null} in INFER mode
     */
    @Override
    public MatrixF backward(MatrixF grads, float learningRate) {
        if (mode == NetworkMode.INFER) {
            return null;
        }

        // Pass a defensive copy to the branch backward: Dropout.backward()
        // modifies its argument in-place, which would corrupt the identity
        // gradient if we passed the original grads reference.
        MatrixF branchGrads = copyOf(grads);
        ListIterator<Layer> it = branch.listIterator(branch.size());
        while (it.hasPrevious()) {
            branchGrads = it.previous().backward(branchGrads, learningRate);
        }

        // dL/dx = grads (identity path) + branchGrads (transformation path)
        return grads.plus(branchGrads);
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /** Creates an independent element-by-element copy of {@code src}. */
    private static MatrixF copyOf(MatrixF src) {
        return src.copy();
    }
}
