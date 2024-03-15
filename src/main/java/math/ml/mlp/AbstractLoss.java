package math.ml.mlp;

import java.util.function.Consumer;
import java.util.function.DoubleConsumer;
import java.util.function.IntFunction;

import net.jamu.matrix.MatrixF;

public class AbstractLoss extends AbstractLayer implements Loss {

    protected Consumer<MatrixF> lossCallback;
    protected DoubleConsumer accuracyCallback;
    protected IntFunction<MatrixF> expectedBatchResultsCallback;

    int batchNumber = 0;

    public AbstractLoss() {
    }

    @Override
    public void registerLossCallback(Consumer<MatrixF> callback) {
        lossCallback = callback;
    }

    @Override
    public void registerAccuracyCallback(DoubleConsumer callback) {
        accuracyCallback = callback;
    }

    @Override
    public void registerBatchExpectedValuesProvider(IntFunction<MatrixF> provider) {
        expectedBatchResultsCallback = provider;
    }

    @Override
    public MatrixF forward(MatrixF prediction) {
        MatrixF expected = null;
        if (expectedBatchResultsCallback == null
                || (expected = expectedBatchResultsCallback.apply(batchNumber)) == null) {
            // there is nothing a Loss function can do without knowing what the expected
            // values for the predictions are
            return null;
        }
        ++batchNumber;
        return expected;
    }
}
