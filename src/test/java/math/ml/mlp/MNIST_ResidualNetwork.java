/*
 * Copyright 2026 Stefan Zobel
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package math.ml.mlp;

import java.util.concurrent.ThreadLocalRandom;

import math.cern.Arithmetic;
import math.ml.loader.MNIST;
import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;
import net.jamu.matrix.Statistics;

/**
 * MNIST classifier demonstrating the use of {@link BatchNorm},
 * {@link ResidualBranch}, and {@link Dropout}.
 *
 * <h2>Network architecture</h2>
 * <pre>
 *  Input (784)
 *      &darr;
 *  Hidden(784 &rarr; 256)
 *  BatchNorm(256)          &larr; normalizes activations across the batch
 *  ReLU
 *  Dropout(0.15)           &larr; fixed mask indexing
 *      &darr;
 *  +-- ResidualBranch --------------------------------+
 *  |   Hidden(256 &rarr; 256)                              |
 *  |   BatchNorm(256)                                 |
 *  |   ReLU                                           |
 *  +-- y = x + F(x) ----------------------------------+
 *      &darr;
 *  Dropout(0.10)
 *      &darr;
 *  Hidden(256 &rarr; 10)
 *  SoftmaxCrossEntropyLoss
 * </pre>
 *
 * <h2>Why BatchNorm before the residual block?</h2>
 * The input to the residual branch and to the identity shortcut must share the
 * same distribution.  Placing BatchNorm before the block ensures both paths
 * see a well-normalized signal.
 *
 * <h2>Why copy-on-entry in ResidualBranch?</h2>
 * {@link Dropout} modifies its input matrix in place.  Without a defensive
 * copy the identity shortcut would receive the already-dropped-out values,
 * corrupting the residual connection.
 */
public class MNIST_ResidualNetwork extends AbstractNetwork {

    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------
    private static final int NUM_LABELS         = 10;
    private static final int BATCH_SIZE         = 200;
    private static final float LOWER            = 0.0f;
    private static final float UPPER            = 1.0f;
    private static final int NUM_EPOCHS         = 50;

    // -----------------------------------------------------------------------
    // Dataset
    // -----------------------------------------------------------------------
    private static final MatrixF IMAGES =
            Statistics.rescaleInplace(
                    MNIST.getTrainingSetImages()
                         .appendMatrix(MNIST.getTrainingSetImagesLeft())
                         .appendMatrix(MNIST.getTrainingSetImagesRight()),
                    LOWER, UPPER);

    private static final MatrixF EXPECT =
            MNIST.getTrainingSetLabels()
                 .appendMatrix(MNIST.getTrainingSetLabels())
                 .appendMatrix(MNIST.getTrainingSetLabels());

    private static final MatrixF TEST_IMAGES =
            Statistics.rescaleInplace(MNIST.getTestSetImages(), LOWER, UPPER);
    private static final MatrixF TEST_EXPECT = MNIST.getTestSetLabels();

    private static final int INPUT_SIZE          = IMAGES.numRows();           // 784
    private static final int NUM_BATCHES_PER_EPOCH = IMAGES.numColumns() / BATCH_SIZE;

    // -----------------------------------------------------------------------
    // Training state
    // -----------------------------------------------------------------------
    private static int    epoch            = 0;
    private static double epochAccuracySum = 0.0;
    private static double epochLossSum     = 0.0;

    // -----------------------------------------------------------------------
    // AbstractNetwork contract
    // -----------------------------------------------------------------------

    @Override
    public void onLossComputationCompleted(MatrixF losses) {
        epochLossSum += Matrices.colsAverage(losses).toScalar();
    }

    @Override
    public void onAccuracyComputationCompleted(double accuracy) {
        epochAccuracySum += accuracy;
    }

    // -----------------------------------------------------------------------
    // Entry point
    // -----------------------------------------------------------------------

    public static void main(String[] args) {

        MNIST_ResidualNetwork net = new MNIST_ResidualNetwork();

        // --- Loss -----------------------------------------------------------
        SoftmaxCrossEntropyLoss loss = new SoftmaxCrossEntropyLoss();
        loss.registerLossCallback(net::onLossComputationCompleted);
        loss.registerAccuracyCallback(net::onAccuracyComputationCompleted);

        // --- Layer 1: linear + BN + ReLU + Dropout --------------------------
        net.add(new Hidden(INPUT_SIZE, 256, "l1", false, true));
        net.add(new BatchNorm(256, "bn1", false, true));   // <- BatchNorm stabilizes training
        net.add(new Relu());
        net.add(new Dropout(0.15f));           // <- fixed Dropout

        // --- Layer 2: residual block (skip connection) ----------------------
        net.add(new ResidualBranch(            // <- ResidualBranch
                new Hidden(256, 256, "res1", false, true),
                new BatchNorm(256, "bn2", false, true),
                new Relu()
        ));
        net.add(new Dropout(0.10f));

        // --- Output layer ---------------------------------------------------
        net.add(new Hidden(256, NUM_LABELS, "out", false, true));
        net.add(loss);

        // -----------------------------------------------------------------------
        // Training loop
        // -----------------------------------------------------------------------
        final float lr = 0.05f;

        long seed = ThreadLocalRandom.current().nextLong();
        Statistics.shuffleColumnsInplace(IMAGES, seed);
        Statistics.shuffleColumnsInplace(EXPECT, seed);

        double maxValidationAccuracy = 0.0;

        for (int epochIdx = 0; epochIdx < NUM_EPOCHS; ++epochIdx) {
            for (int b = 0; b < NUM_BATCHES_PER_EPOCH; ++b) {
                int startCol = b * BATCH_SIZE;
                MatrixF input = IMAGES.selectConsecutiveColumns(startCol, startCol + BATCH_SIZE - 1);
                MatrixF expected = EXPECT.selectConsecutiveColumns(startCol, startCol + BATCH_SIZE - 1);
                net.train(input, expected, lr);
            }

            double trainingAccuracy  = Arithmetic.round(epochAccuracySum / NUM_BATCHES_PER_EPOCH, 6);
            double avgTrainingLoss   = Arithmetic.round(epochLossSum     / NUM_BATCHES_PER_EPOCH, 6);
            double validationAccuracy = net.validationAccuracy();
            // keep-best: only the improved model reaches the disk
            if (validationAccuracy > maxValidationAccuracy) {
                maxValidationAccuracy = validationAccuracy;
                net.storeParameters();
            }

            System.out.println("epoch " + epoch
                    + "   train acc: " + trainingAccuracy
                    + "   train loss: " + avgTrainingLoss
                    + "   val acc: " + validationAccuracy
                    + "   max val acc: " + maxValidationAccuracy);

            epochAccuracySum = 0.0;
            epochLossSum     = 0.0;
            ++epoch;

            if (validationAccuracy >= 0.99) {
                System.out.println("Reached 99 % validation accuracy. Stopping.");
                break;
            }
            if (epoch > 5 && validationAccuracy < trainingAccuracy - 0.05) {
                System.out.println("Potential overfitting. Stopping.");
                break;
            }

            // reshuffle between epochs
            seed = ThreadLocalRandom.current().nextLong();
            Statistics.shuffleColumnsInplace(IMAGES, seed);
            Statistics.shuffleColumnsInplace(EXPECT, seed);
        }

        System.out.println("\nFinal validation accuracy: " + net.validationAccuracy());
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    private double validationAccuracy() {
        MatrixF predict = infer(TEST_IMAGES);
        return CategorialAccuracy.computeAccuracy(predict, TEST_EXPECT);
    }
}
