package math.ml.mlp;

import net.jamu.matrix.MatrixF;

public interface Network {

    MatrixF infer(MatrixF input);

}
