package math.ml.mlp;

import net.jamu.matrix.MatrixF;

public abstract class AbstractLayer implements Layer {

    protected MatrixF input;
    protected NetworkMode mode = NetworkMode.INFER;

    @Override
    public MatrixF forward(MatrixF input) {
        if (mode == NetworkMode.TRAIN) {
            // i x m
            this.input = input;
        }
        return null;
    }

    @Override
    public void setMode(NetworkMode mode) {
        this.mode = mode;
    }
}
