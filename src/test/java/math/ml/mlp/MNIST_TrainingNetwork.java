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
public class MNIST_TrainingNetwork extends AbstractNetwork {

    public MNIST_TrainingNetwork() {
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
    // 784 x 180_000
    private static final MatrixF IMAGES = Statistics.zscoreColumnsInplace(MNIST.getTrainingSetImages()
            .appendMatrix(MNIST.getTrainingSetImagesLeft()).appendMatrix(MNIST.getTrainingSetImagesRight()));

    // 10 x 180_000
    private static final MatrixF EXPECT = MNIST.getTrainingSetLabels().appendMatrix(MNIST.getTrainingSetLabels())
            .appendMatrix(MNIST.getTrainingSetLabels());

    private static final MatrixF TEST_IMAGES = Statistics.zscoreColumnsInplace(MNIST.getTestSetImages());
    private static final MatrixF TEST_EXPECT = MNIST.getTestSetLabels();

    private static final int INPUT_SIZE = IMAGES.numRows();
    private static final int NUM_BATCHES_PER_EPOCH = IMAGES.numColumns() / BATCH_SIZE;
    private static final int NUM_BATCHES = 100;
    private static int epoch = 0;
    private static double epochAccuraciesSum = 0.0;
    private static double epochLossesSum = 0.0;

    public static void main(String[] args) {
        MNIST_TrainingNetwork net = new MNIST_TrainingNetwork();
//        CrossEntropyLoss loss = new CrossEntropyLoss(); // XXX
        SoftmaxCrossEntropyLoss loss = new SoftmaxCrossEntropyLoss();
        loss.registerAccuracyCallback(net::onAccuracyComputationCompleted);
        loss.registerLossCallback(net::onLossComputationCompleted);
        loss.registerBatchExpectedValuesProvider(net::getExpectedBatchResults);

        net.add(new Hidden(INPUT_SIZE, 768, "layer1", false, true));
        net.add(new Relu()); // 768
        net.add(new Hidden(768, 384, "layer2", false, true));
        net.add(new Relu()); // 384
        net.add(new Hidden(384, 256, "layer3", false, true));
        net.add(new Relu()); // 256
        net.add(new Hidden(256, NUM_LABELS, "layer4", false, true));
        net.add(new Relu()); // 10
//        net.add(new Softmax()); // XXX
        net.add(loss);

        final float learningRate = 0.001f; // XXX ?

        // shuffle images and labels randomly
        long seed = ThreadLocalRandom.current().nextLong();
        Statistics.shuffleColumnsInplace(IMAGES, seed);
        Statistics.shuffleColumnsInplace(EXPECT, seed);

        // train for up to 100 epochs
        for (int i = 0; i <= NUM_BATCHES * NUM_BATCHES_PER_EPOCH; ++i) {
            int startCol = getStartColumn(i);
            MatrixF input = IMAGES.selectConsecutiveColumns(startCol, startCol + BATCH_SIZE - 1);
            net.train(input, learningRate);
            if (i > 0 && (i % NUM_BATCHES_PER_EPOCH == 0)) {
                double trainingAccuracy = Arithmetic.round(epochAccuraciesSum / NUM_BATCHES_PER_EPOCH, 6);
                double validationAccuracy = net.validationAccuracy();
                System.out.println("epoch " + epoch + "   : avg. accuracy: " + trainingAccuracy + "   : avg. loss: "
                        + Arithmetic.round(epochLossesSum / NUM_BATCHES_PER_EPOCH, 6)
                        + "   : validation avg. accuracy: " + validationAccuracy);
                epochAccuraciesSum = 0.0;
                epochLossesSum = 0.0;
                ++epoch;
                if (validationAccuracy < trainingAccuracy) {
                    System.out.println("potential overfitting. BREAK.");
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
