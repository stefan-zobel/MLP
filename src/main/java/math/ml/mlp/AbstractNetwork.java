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

import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;
import java.util.Objects;

import net.jamu.matrix.MatrixF;

/** A network built as a flat list of layers, the last one a {@link Loss}. */
public abstract class AbstractNetwork implements TrainableNetwork {

    /** The layers in forward order. */
    protected ArrayList<Layer> layers = new ArrayList<>();

    /** Number of batches trained so far. */
    protected int batchCount = 0;

    /** Creates an empty network; add the layers with {@link #add(Layer)}. */
    public AbstractNetwork() {
    }

    /**
     * Whether {@link #train} has to protect the caller's matrix. Any layer, not just
     * the first: a pass-through layer hands its argument straight on.
     */
    private boolean copyInput = false;

    /** Every parameter of every layer, in the order the layers were added. */
    private final ArrayList<Parameter> parameters = new ArrayList<>();

    private Optimizer optimizer;

    @Override
    public Network add(Layer layer) {
        layers.add(layer);
        copyInput |= layer.mutatesInput();
        List<Parameter> own = layer.parameters();
        parameters.addAll(own);
        if (optimizer != null) {
            // registering as we go and registering retroactively below means the order of
            // add() and optimizer() does not matter, and each parameter is seen once
            for (Parameter p : own) {
                optimizer.add(p);
            }
        }
        return this;
    }

    @Override
    public Network optimizer(Optimizer optimizer) {
        if (this.optimizer != null) {
            throw new IllegalStateException("the optimizer of this network is already set");
        }
        this.optimizer = Objects.requireNonNull(optimizer, "optimizer");
        for (Parameter p : parameters) {
            optimizer.add(p);
        }
        return this;
    }

    @Override
    public Network train(MatrixF input, MatrixF expected) {
        if (expected == null) {
            throw new IllegalArgumentException("expected values must not be null");
        }
        if (layers.size() < 2 || !(layers.get(layers.size() - 1) instanceof Loss lossLayer)) {
            throw new IllegalStateException(
                    "training needs at least two layers and the last one must be a Loss");
        }
        // after the checks above, so that a malformed layer list is still reported as
        // such. A network without parameters has nothing for an optimizer to do.
        if (optimizer == null && !parameters.isEmpty()) {
            throw new IllegalStateException("this network has trainable parameters but no optimizer");
        }
        // hand the targets to the loss for exactly this batch
        lossLayer.setExpectedValues(expected);
        if (copyInput) {
            // a layer in this net writes into what it is given, and the caller's
            // matrix is usually a batch it wants to reuse. infer() needs no such
            // guard: no layer mutates in INFER mode.
            input = input.copy();
        }
        for (Layer layer : layers) {
            layer.setMode(NetworkMode.TRAIN);
            input = layer.forward(input);
        }
        // input is now the output from the last layer which is the loss function, thus
        // it holds the gradient of the loss function. Now do the back-propagation.
        ListIterator<Layer> it = layers.listIterator(layers.size());
        while (it.hasPrevious()) {
            Layer layer = it.previous();
            if (layer instanceof Loss loss && !loss.producesPredictionInInferMode()) {
                // a plain Loss returns the gradient from its forward() method, its
                // backward() method does nothing. A fused loss keeps the gradient and
                // hands it back from backward(), so it stays in the chain.
                continue;
            }
            // propagate the gradients backwards to the previous layer
            input = layer.backward(input);
        }
        // every gradient of this batch now exists, so the parameters can move
        if (optimizer != null) {
            optimizer.step();
        }
        ++batchCount;
        return this;
    }

    @Override
    public MatrixF infer(MatrixF input) {
        for (Layer layer : layers) {
            layer.setMode(NetworkMode.INFER);
            if (layer instanceof Loss loss && !loss.producesPredictionInInferMode()) {
                // a plain Loss would return the gradient from its forward() method which is
                // not a prediction, also it may call callbacks which might not have a
                // sensible implementation if we are doing inference only, so skip this.
                // A fused loss is an exception as it applies its output activation itself.
                // We assume that a Loss, if there is any, is always the last layer
                break;
            }
            input = layer.forward(input);
        }
        // this is the prediction of the last layer
        return input;
    }

    /**
     * Persists the parameters of every layer that was constructed with storing
     * enabled; layers without storable parameters do nothing.
     *
     * <p>Call this explicitly from the training loop, typically only when the
     * validation score improved. Inference deliberately does not persist.
     */
    public void storeParameters() {
        for (Layer layer : layers) {
            layer.storeParameters();
        }
    }
}
