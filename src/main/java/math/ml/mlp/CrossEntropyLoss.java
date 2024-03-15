package math.ml.mlp;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

public class CrossEntropyLoss extends AbstractLoss {

    public CrossEntropyLoss() {
    }

    // compute losses
    // compute accuracy
    // compute gradients
    // return gradients
    @Override
    public MatrixF forward(MatrixF prediction) {
        MatrixF expected = getExpectation();
        if (expected == null) {
            return null;
        }
        computeLosses(prediction, expected);
        computeAccuracy(prediction, expected);
        return computeGradients(prediction, expected);
    }

    // TODO: can we express this as a matrix operation?
    private MatrixF computeGradients(MatrixF pred, MatrixF expect) {
        MatrixF gradients = Matrices.createF(pred.numRows(), pred.numColumns());
        for (int i = 0, colEnd = pred.endCol(), rowEnd = pred.endRow(); i <= colEnd; ++i) {
            for (int j = 0; j <= rowEnd; ++j) {
                float expected = expect.getUnsafe(j, i);
                if (expected != 0.0f) {
                    gradients.set(j, i, (-expected / clamp(pred.getUnsafe(j, i))));
                }
            }
        }
        return gradients;
    }

    // TODO: can we express this as a matrix operation?
    /**
     * Computes 1 x batchSize row vector of cross-entropy losses for same-sized
     * matrices of expected values (expect) and predicted values (pred) expressed as
     * column vectors in those matrices.
     * 
     * @param pred   predicted values as a matrix of column vectors
     * @param expect expected values as a matrix of column vectors
     */
    private void computeLosses(MatrixF pred, MatrixF expect) {
        if (lossCallback != null) {
            MatrixF loss = Matrices.createF(1, pred.numColumns());
            for (int i = 0, colEnd = pred.endCol(), rowEnd = pred.endRow(); i <= colEnd; ++i) {
                float productSum = 0.0f;
                for (int j = 0; j <= rowEnd; ++j) {
                    float expected = expect.getUnsafe(j, i);
                    if (expected != 0.0f) {
                        productSum += expected * log(pred.getUnsafe(j, i));
                    }
                }
                loss.setUnsafe(0, i, -productSum);
            }
            lossCallback.accept(loss);
        }
    }

    private void computeAccuracy(MatrixF pred, MatrixF expect) {
        if (accuracyCallback != null) {
            // TODO
        }
    }

    private static float log(float x) {
        return (float) Math.log(clamp(x));
    }

    private static float clamp(float x) {
        // x should never be <= 0
        if (x <= 0.0f) {
            return Float.MIN_NORMAL;
        }
        return x;
    }
}
