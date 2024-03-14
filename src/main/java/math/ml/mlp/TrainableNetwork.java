package math.ml.mlp;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

public interface TrainableNetwork extends Network {

    // this NEEDS to be implemented in order for a Network to be trainable
    Network train(MatrixF input, float learningRate);

    // this NEEDS to be implemented in order for a Network to be trainable
    MatrixF getExpectedBatchResults(int batchNumber);

    default Network add(Layer layer) {
        // not every Network needs to be constructed layer by layer
        return this;
    }

    default void onLossComputationCompleted(MatrixF losses) {
        System.out.println("Avg. loss: " + Matrices.colsAverage(losses).toScalar());
    }

    default void onAccuracyComputationCompleted(double accuracy) {
        System.out.println("Accuracy: " + accuracy);
    }
}
