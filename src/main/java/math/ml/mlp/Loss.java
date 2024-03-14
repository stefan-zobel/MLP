package math.ml.mlp;

import java.util.function.Consumer;
import java.util.function.DoubleConsumer;
import java.util.function.IntFunction;

import net.jamu.matrix.MatrixF;

public interface Loss extends Layer {

    void registerLossCallback(Consumer<MatrixF> callback);

    void registerAccuracyCallback(DoubleConsumer callback);

    void registerBatchExpectedValuesProvider(IntFunction<MatrixF> provider);

    // by default backward() for a Loss function does nothing and shouldn't be
    // called
    default MatrixF backward(MatrixF unused1, float unused2) {
        return null;
    }
}
