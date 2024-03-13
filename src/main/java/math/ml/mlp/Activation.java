package math.ml.mlp;

import math.dl.GELU;
import net.jamu.matrix.FFunction;
import net.jamu.matrix.MatrixF;

public class Activation extends AbstractLayer {

    FFunction fun = GELU::geluF; // TODO
    FFunction deriv = GELU::dgeluF_dx; // TODO

    public Activation() {
        // TODO
    }

    @Override
    public MatrixF forward(MatrixF input) {
        // j x m
        super.forward(input);
        return input.map(fun);
    }

    // outputGrads : j x m
    @Override
    public MatrixF backward(MatrixF outputGrads, float unused) {
        if (mode == NetworkMode.INFER) {
            return null;
        }
        // (j x m) o (j x m)
        MatrixF out = outputGrads.hadamard(input.mapInplace(deriv));
        input = null;
        return out;
    }
}
