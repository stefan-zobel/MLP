/*
 * Copyright 2024 Stefan Zobel
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
 * A simple MLP for MNIST.
 */
public class MNIST_TrainingNetwork2 extends AbstractNetwork {

    public MNIST_TrainingNetwork2() {
    }

    @Override
    public void onLossComputationCompleted(MatrixF losses) {
        epochLossesSum += Matrices.colsAverage(losses).toScalar();
    }

    @Override
    public void onAccuracyComputationCompleted(double accuracy) {
        epochAccuraciesSum += accuracy;
    }

    private static int getStartColumn(int batchNumber) {
        batchNumber = batchNumber % NUM_BATCHES_PER_EPOCH;
        return batchNumber * BATCH_SIZE;
    }

    @Override
    public MatrixF getExpectedBatchResults(int batchNumber) {
        int col = getStartColumn(batchNumber);
        return EXPECT.selectConsecutiveColumns(col, col + BATCH_SIZE - 1);
    }

    private static final int NUM_LABELS = 10;
    private static final int BATCH_SIZE = 200;
    private static final float LOWER = 0.0f;
    private static final float UPPER = 1.0f;
    // 784 x 180_000
    private static final MatrixF IMAGES = Statistics.rescaleInplace(MNIST.getTrainingSetImages()
            .appendMatrix(MNIST.getTrainingSetImagesLeft()).appendMatrix(MNIST.getTrainingSetImagesRight()), LOWER,
            UPPER);

    // 10 x 180_000
    private static final MatrixF EXPECT = MNIST.getTrainingSetLabels().appendMatrix(MNIST.getTrainingSetLabels())
            .appendMatrix(MNIST.getTrainingSetLabels());

    private static final MatrixF TEST_IMAGES = Statistics.rescaleInplace(MNIST.getTestSetImages(), LOWER, UPPER);
    private static final MatrixF TEST_EXPECT = MNIST.getTestSetLabels();

    private static final int INPUT_SIZE = IMAGES.numRows();
    private static final int NUM_BATCHES_PER_EPOCH = IMAGES.numColumns() / BATCH_SIZE;
    private static final int NUM_EPOCHS = 200;
    private static int epoch = 0;
    private static double epochAccuraciesSum = 0.0;
    private static double epochLossesSum = 0.0;

    public static void main(String[] args) {
        final float dropoutRate = 0.080f; // XXX

        MNIST_TrainingNetwork2 net = new MNIST_TrainingNetwork2();
        SoftmaxCrossEntropyLoss loss = new SoftmaxCrossEntropyLoss();
        loss.registerAccuracyCallback(net::onAccuracyComputationCompleted);
        loss.registerLossCallback(net::onLossComputationCompleted);
        loss.registerBatchExpectedValuesProvider(net::getExpectedBatchResults);

        net.add(new Hidden(INPUT_SIZE, 768, "layer1", false, true));
        net.add(new Dropout(dropoutRate / 3)); // / 5 / 3
        net.add(new Relu()); // 768
        net.add(new Hidden(768, 384, "layer2", false, true));
        net.add(new Dropout(dropoutRate)); // / 4 / 2
        net.add(new Relu()); // 384
        net.add(new Hidden(384, 256, "layer3", false, true));
        net.add(new Dropout(dropoutRate)); // / 2
        net.add(new Relu()); // 256
        net.add(new Hidden(256, NUM_LABELS, "layer4", false, true));
        net.add(new Dropout(dropoutRate * 1.25f));
        net.add(new Relu()); // 10
        net.add(loss);

        final float learningRate = 0.5f; // XXX

        // shuffle images and labels randomly
        long seed = ThreadLocalRandom.current().nextLong();
        Statistics.shuffleColumnsInplace(IMAGES, seed);
        Statistics.shuffleColumnsInplace(EXPECT, seed);

        double maxValidationAccuracy = 0.0;

        // train for up to NUM_EPOCHS epochs
        for (int i = 0; i <= NUM_EPOCHS * NUM_BATCHES_PER_EPOCH; ++i) {
            int startCol = getStartColumn(i);
            MatrixF input = IMAGES.selectConsecutiveColumns(startCol, startCol + BATCH_SIZE - 1);
            net.train(input, learningRate);
            if (i > 0 && (i % NUM_BATCHES_PER_EPOCH == 0)) {
                double trainingAccuracy = Arithmetic.round(epochAccuraciesSum / NUM_BATCHES_PER_EPOCH, 6);
                double avgTrainingLoss = Arithmetic.round(epochLossesSum / NUM_BATCHES_PER_EPOCH, 6);
                double validationAccuracy = net.validationAccuracy();
                maxValidationAccuracy = Math.max(maxValidationAccuracy, validationAccuracy);
                System.out.println("epoch " + epoch + "   : avg. accuracy: " + trainingAccuracy + "   : avg. loss: "
                        + avgTrainingLoss + "   : validation avg. accuracy: " + validationAccuracy + "   : max acc.: "
                        + maxValidationAccuracy);
                epochAccuraciesSum = 0.0;
                epochLossesSum = 0.0;
                ++epoch;
                if (validationAccuracy < trainingAccuracy) {
                    System.out.println("potential overfitting. Stopping.");
                    break;
                }
                if (validationAccuracy >= 0.99) {
                    System.out.println("Found a net with accuracy " + validationAccuracy + ". Stopping.");
                    break;
                }
                // reshuffle before the next epoch
                seed = ThreadLocalRandom.current().nextLong();
                Statistics.shuffleColumnsInplace(IMAGES, seed);
                Statistics.shuffleColumnsInplace(EXPECT, seed);
            }
        }

        System.out.println("\nDone with training. Checking last validation accuracy.");
        double accuracy = net.validationAccuracy();
        System.out.println("validation : avg. accuracy in validation: " + accuracy);
    }

    private double validationAccuracy() {
        MatrixF predict = infer(TEST_IMAGES);
        return CategorialAccuracy.computeAccuracy(predict, TEST_EXPECT);
    }
}
