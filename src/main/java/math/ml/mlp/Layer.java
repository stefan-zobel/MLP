package math.ml.mlp;

import net.jamu.matrix.MatrixF;

public interface Layer {

    /**
     * Forward pass.
     * 
     * @param input the forward input into this layer
     * @return the forward output of this layer
     */
    MatrixF forward(MatrixF input);

    /**
     * Backward pass.
     * 
     * @param grads error gradients with respect to the output of this layer
     * @param learningRate the learning rate ({@code 0 < r < 1})
     * @return error gradients with respect to the input of this layer
     */
    MatrixF backward(MatrixF grads, float learningRate);

    Layer nextLayer();

    MatrixF input();
}
