package math.ml.mlp;

import net.jamu.matrix.MatrixF;

public abstract class AbstractLayer implements Layer {

    protected MatrixF input;
    protected NetworkMode mode = NetworkMode.TRAIN; // TODO
    private Layer next;

    @Override
    public MatrixF forward(MatrixF input) {
        if (mode == NetworkMode.TRAIN) {
            // i x m
            this.input = input;
        }
        return null;
    }

    @Override
    public Layer nextLayer() {
        return next;
    }

    @Override
    public MatrixF input() {
        return input;
    }
}
