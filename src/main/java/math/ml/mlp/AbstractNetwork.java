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
import java.util.ListIterator;

import net.jamu.matrix.MatrixF;

public abstract class AbstractNetwork implements TrainableNetwork {

    protected ArrayList<Layer> layers = new ArrayList<>();

    protected int batchCount = 0;

    public AbstractNetwork() {
    }

    // this NEEDS to be implemented for the Network TRAIN mode!
    public abstract MatrixF getExpectedBatchResults(int batchNumber);

    @Override
    public Network add(Layer layer) {
        layers.add(layer);
        return this;
    }

    @Override
    public Network train(MatrixF input, float learningRate) {
        if (layers.size() < 2 || !(layers.get(layers.size() - 1) instanceof Loss)) {
            // training requires at least two layers and the last one must be a loss
            // function
            return null;
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
            if (layer instanceof Loss && !(layer instanceof SoftmaxCrossEntropyLoss)) {
                // a Loss returns the gradient from its forward() method, its backward() method
                // does nothing
                continue;
            }
            // propagate the gradients backwards to the previous layer
            input = layer.backward(input, learningRate);
        }
        ++batchCount;
        return this;
    }

    @Override
    public MatrixF infer(MatrixF input) {
        for (Layer layer : layers) {
            layer.setMode(NetworkMode.INFER);
            if (layer instanceof Loss && !(layer instanceof SoftmaxCrossEntropyLoss)) {
                // a Loss would return the gradient from its forward() method which is not a
                // prediction, also it may call callbacks which might not have a sensible
                // implementation if we are doing inference only, so skip this.
                // SoftmaxCrossEntropyLoss is an exception as it behaves like Softmax in INFER
                // mode. We assume that a Loss, if there is any, is always the last layer
                break;
            }
            input = layer.forward(input);
        }
        for (Layer layer : layers) {
            if (layer instanceof Hidden) {
                ((Hidden) layer).storeWeights();
                ((Hidden) layer).storeBiases();
            }
        }
        // this is the prediction of the last layer
        return input;
    }
}
