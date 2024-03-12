package math.ml.mlp;

import net.jamu.matrix.MatrixF;

public interface Layer {

    MatrixF forward(MatrixF input);

    /**
     * 
     * @param grads error gradients with respect to the output of this layer
     * @param learningRate
     * @return error gradients with respect to the input of this layer
     */
    MatrixF backward(MatrixF grads, float learningRate);

    Layer nextLayer();

    MatrixF input();
}
