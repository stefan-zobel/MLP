package math.ml.mlp;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

public class Hidden extends AbstractLayer {

    // j x i
    MatrixF weights;
    // j x 1
    MatrixF biases;

    public Hidden() {
        // TODO ...
    }

    @Override
    public MatrixF forward(MatrixF input) {
        if (mode == NetworkMode.TRAIN) {
            // i x m
            this.input = input;
        }
        return weights.times(input).addBroadcastedVectorInplace(biases);
    }

    // outputGrads : j x m
    @Override
    public MatrixF backward(MatrixF outputGrads, float learningRate) {
        if (mode == NetworkMode.INFER) {
            return null;
        }
        // (i x j) * (j x m) = (i x m)
        MatrixF inputErrJacobian = weights.transposedTimes(outputGrads);
        // (j x i) = (j * m) * (m x i)
        // we don't have a single y1, but m exemplars of it, just as with y2 etc.
        //
        // dE/dW =
        // dE/dw11 dE/dw12 ... dE/dw1i
        // ...
        // dE/dwj1 dE/dwj2 --- dE/dwji
        //
        // generally: dE/dwji = dE/dyj * xi
        // => dE/dW = dE/dY * X_trans
        // the OutputGrads:
        // dE1/dY11 dE2/dY12 ... dEm/dY1m
        // ....
        // dE1/dYj1 dE2/dYj2 ... dEm/dYjm
        // the Inputs (transposed):
        // X11 X12 .......... X1i
        // X21 X22 .......... X2i
        // ...
        // Xm1 Xm2 .......... Xmi
        // ----
        // dE/dw11 expanded:
        // dE1/dY11 * X11 + dE2/dY12 * X21 + ... dEm/dY1m * Xm1 == dE/dw11
        // batch size = m = outputGrads.numColumns()
        MatrixF avgWeightsGrad = outputGrads.timesTransposed(input).scaleInplace(1.0f / outputGrads.numColumns());
        input = null;
        // j x 1
        MatrixF avgBiasesGrad = Matrices.colsAverage(outputGrads);
        weights.addInplace(-learningRate, avgWeightsGrad);
        biases.addInplace(-learningRate, avgBiasesGrad);
        return inputErrJacobian;
    }

}
