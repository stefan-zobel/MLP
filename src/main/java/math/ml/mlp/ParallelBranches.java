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

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

/**
 * A composite {@link Layer} that processes its input through multiple
 * independent branch sequences <em>in parallel</em> and therefore enables
 * fork/merge topologies inside the otherwise strictly sequential
 * {@link AbstractNetwork}.
 *
 * <p><b>Forward pass:</b> the same input matrix is fed into every branch.
 * Because some layers (e.g. {@link Dropout}) modify their input in-place,
 * each branch receives its own defensive copy of the original input, into a
 * buffer that is reused across steps. The resulting output matrices are then
 * vertically stacked (row-wise concatenated) in branch order: if branch
 * {@code i} produces an output of shape {@code k_i x m}, the combined forward
 * output has shape {@code (sum k_i) x m}.
 *
 * <p><b>Backward pass:</b> the incoming gradient is split vertically
 * according to the per-branch output row counts recorded during the last
 * forward pass. Each gradient slice is back-propagated through its own
 * branch in reverse layer order. The resulting input-side gradients are
 * then <em>summed element-wise</em>: because the same input tensor feeds all
 * branches, the chain rule gives
 * {@code dL/dx = sum_i (dL/dy_i * dy_i/dx)}, i.e. the contributions
 * are additive.
 *
 * <p><b>Mode propagation:</b> {@link #setMode} is overridden to propagate
 * the network mode ({@code TRAIN} / {@code INFER}) to every layer in every
 * branch, so the outer training loop does not need to be aware of the
 * branching structure.
 *
 * <p><b>Parameter persistence:</b> {@link #storeParameters} is overridden to
 * delegate to all sub-layers in all branches.
 *
 * <h2>Example usage in a variational autoencoder</h2>
 * <pre>{@code
 * // After the shared encoder the path splits into a "mu"- and a "logvar"-head:
 * net.add(new ParallelBranches(
 *     List.of(new Hidden(128, 16, "mu")),       // mu  branch rows [0..15]
 *     List.of(new Hidden(128, 16, "logvar"))    // logvar branch rows [16..31]
 * ));
 * // Returns (32 x m). The next layer is usually a VAEReparamLayer(16).
 * }</pre>
 */
public class ParallelBranches extends AbstractLayer {

    private final List<List<Layer>> branches;

    /**
     * Number of output rows produced by each branch during the most recent
     * forward pass.  Used to split the incoming gradient in backward().
     */
    private final int[] branchOutputRows;

    /**
     * Whether a branch needs its own copy of the shared input. Not the same
     * question as {@link #mutatesInput()}: this layer never writes into its own
     * argument, it only has to keep one branch from writing into what the other
     * branches still have to read.
     */
    private final boolean copyInput;

    /**
     * Reused across steps rather than allocated per call: the stacked output, one gradient
     * slice per branch, the accumulated input gradient, and one input copy per branch when
     * {@link #copyInput} is set. They are separate buffers on purpose -- a numerical gradient
     * check holds the matrix backward returns while it calls forward hundreds of times, so
     * nothing forward writes into may be what backward returned.
     */
    private final MatrixF[] gradSlices;
    private final MatrixF[] inputCopies;
    private MatrixF output;
    private MatrixF gradientsOut;

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    /**
     * Convenience varargs constructor.
     *
     * @param branches one or more branch sequences; each sequence is a
     *                 {@code List<Layer>} that will be executed in order
     */
    @SafeVarargs
    public ParallelBranches(List<Layer>... branches) {
        this(Arrays.asList(branches));
    }

    /**
     * Constructor that accepts a pre-built list of branch sequences.
     *
     * @param branches list of branch sequences
     * @throws IllegalArgumentException if the list is empty
     */
    public ParallelBranches(List<List<Layer>> branches) {
        if (branches.isEmpty()) {
            // there is nothing this layer could return: a matrix needs at least one row
            throw new IllegalArgumentException("a ParallelBranches needs at least one branch");
        }
        this.branches = new ArrayList<>(branches);
        this.branchOutputRows = new int[branches.size()];
        boolean mutates = false;
        for (List<Layer> branch : this.branches) {
            for (Layer layer : branch) {
                mutates |= layer.mutatesInput();
            }
        }
        this.copyInput = mutates;
        this.gradSlices = new MatrixF[branches.size()];
        this.inputCopies = mutates ? new MatrixF[branches.size()] : null;
    }

    // -------------------------------------------------------------------------
    // Layer contract
    // -------------------------------------------------------------------------

    /**
     * Propagates the network mode to every layer in every branch so that the
     * outer training loop does not need to know about the internal structure.
     */
    @Override
    public void setMode(NetworkMode mode) {
        super.setMode(mode); // sets this.mode used in backward()
        for (List<Layer> branch : branches) {
            for (Layer layer : branch) {
                layer.setMode(mode);
            }
        }
    }

    /**
     * Delegates parameter persistence to every layer in every branch.
     */
    @Override
    public void storeParameters() {
        for (List<Layer> branch : branches) {
            for (Layer layer : branch) {
                layer.storeParameters();
            }
        }
    }

    /** Collects the parameters of every layer in every branch, in branch order. */
    @Override
    public List<Parameter> parameters() {
        List<Parameter> all = new ArrayList<>();
        for (List<Layer> branch : branches) {
            for (Layer layer : branch) {
                all.addAll(layer.parameters());
            }
        }
        return all;
    }

    /**
     * Runs each branch with the same (independently copied) input and returns
     * the vertically stacked output.
     *
     * <p>Note: {@code super.forward(input)} is intentionally <em>not</em>
     * called here because {@code ParallelBranches} has no own learnable
     * parameters and does not need to cache the input itself; each branch
     * layer caches its own input as required.
     *
     * @param input the shared input matrix ({@code n x m})
     * @return vertically stacked branch outputs ({@code (sum k_i) x m})
     */
    @Override
    public MatrixF forward(MatrixF input) {
        int n = branches.size();
        MatrixF[] outputs = new MatrixF[n];
        int totalRows = 0;
        for (int b = 0; b < n; b++) {
            // Give every branch its own copy of the input so that in-place
            // operations (e.g. Dropout) in one branch do not corrupt the
            // others. Not needed when no branch layer writes into its input.
            MatrixF x = input;
            if (copyInput) {
                inputCopies[b] = ensure(inputCopies[b], input.numRows(), input.numColumns());
                x = inputCopies[b].setInplace(input);
            }
            for (Layer layer : branches.get(b)) {
                x = layer.forward(x);
            }
            outputs[b] = x;
            branchOutputRows[b] = x.numRows();
            totalRows += x.numRows();
        }
        // Every branch output is held until here, which is what keeps a mutating layer in one
        // branch from being visible in another.
        int cols = outputs[0].numColumns();
        output = ensure(output, totalRows, cols);
        int rowOffset = 0;
        for (MatrixF branchOut : outputs) {
            output.setSubmatrixInplace(rowOffset, 0, branchOut, 0, 0, branchOut.numRows() - 1, cols - 1);
            rowOffset += branchOut.numRows();
        }
        return output;
    }

    /**
     * Splits the incoming gradient, back-propagates each slice through its
     * branch in reverse order, and returns the element-wise sum of the
     * resulting input-side gradients.
     *
     * @param grads vertically stacked gradients ({@code (sum k_i) x m})
     * @return summed input-side gradients ({@code n x m}); {@code null} in
     *         {@code INFER} mode
     */
    @Override
    public MatrixF backward(MatrixF grads) {
        if (mode == NetworkMode.INFER) {
            return null;
        }
        int n = branches.size();
        int cols = grads.numColumns();
        MatrixF summedInputGrads = null;
        int rowOffset = 0;
        for (int b = 0; b < n; b++) {
            int rows = branchOutputRows[b];
            // Extract the gradient slice that belongs to this branch. One buffer per branch,
            // because a branch that writes into the gradient it is handed must not be able to
            // reach what another branch still has to read.
            gradSlices[b] = ensure(gradSlices[b], rows, cols);
            MatrixF branchGrads = gradSlices[b].setSubmatrixInplace(0, 0, grads, rowOffset, 0,
                    rowOffset + rows - 1, cols - 1);
            rowOffset += rows;
            // Back-propagate through the branch layers in reverse order.
            MatrixF g = branchGrads;
            List<Layer> branch = branches.get(b);
            ListIterator<Layer> it = branch.listIterator(branch.size());
            while (it.hasPrevious()) {
                g = it.previous().backward(g);
            }
            // Accumulate (chain rule: same input -> additive gradient terms).
            if (summedInputGrads == null) {
                // Unconditional, and not covered by mutatesGradients(): the property
                // here is that a branch may still hold a reference to the matrix it
                // returned, which addInplace below would overwrite.
                gradientsOut = ensure(gradientsOut, g.numRows(), g.numColumns());
                summedInputGrads = gradientsOut.setInplace(g);
            } else {
                summedInputGrads.addInplace(1.0f, g);
            }
        }
        return summedInputGrads;
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /**
     * Returns {@code buffer} when it already has the wanted shape, and a fresh matrix of that
     * shape otherwise.
     */
    private static MatrixF ensure(MatrixF buffer, int rows, int columns) {
        if (buffer == null || buffer.numRows() != rows || buffer.numColumns() != columns) {
            return Matrices.createF(rows, columns);
        }
        return buffer;
    }

}
