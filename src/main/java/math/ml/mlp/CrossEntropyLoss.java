package math.ml.mlp;

import net.jamu.matrix.MatrixF;

public class CrossEntropyLoss extends AbstractLoss {

    public CrossEntropyLoss() {
        // TODO
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
        // TODO
        return null;
    }
}
