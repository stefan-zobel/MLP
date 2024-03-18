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

import math.ml.loader.MNIST;
import net.jamu.matrix.MatrixF;
import net.jamu.matrix.Statistics;

/**
 * A simple MLP for MNIST.
 */
public class MNIST_TrainingNetwork extends AbstractNetwork {

    public MNIST_TrainingNetwork() {
    }

    public void onAccuracyComputationCompleted(double accuracy) {
        epochAccuraciesSum += accuracy;
    }

    private static int getStartColumn(int batchNumber) {
        batchNumber = batchNumber % NUM_BATCHES_PER_EPOCH;
        return batchNumber * BATCH_SIZE;
    }

    public MatrixF getExpectedBatchResults(int batchNumber) {
        int col = getStartColumn(batchNumber);
        return EXPECT.selectConsecutiveColumns(col, col + BATCH_SIZE - 1);
    }

    private static final int NUM_LABELS = 10;
    private static final int BATCH_SIZE = 200;
    // 784 x 60_000
    private static final MatrixF IMAGES = Statistics.zscoreInplace(MNIST.getTrainingSetImages());
    // 10 x 60_000
    private static final MatrixF EXPECT = MNIST.getTrainingSetLabels();
    private static final int INPUT_SIZE = IMAGES.numRows();
    private static final int NUM_BATCHES_PER_EPOCH = IMAGES.numColumns() / BATCH_SIZE;
    private static final int NUM_BATCHES = 40;
    private static int epoch = 0;
    private static double epochAccuraciesSum = 0.0;

    public static void main(String[] args) {
        MNIST_TrainingNetwork net = new MNIST_TrainingNetwork();
//        CrossEntropyLoss loss = new CrossEntropyLoss(); // XXX
        SoftmaxCrossEntropyLoss loss = new SoftmaxCrossEntropyLoss();
        loss.registerAccuracyCallback(net::onAccuracyComputationCompleted);
        loss.registerBatchExpectedValuesProvider(net::getExpectedBatchResults);

        net.add(new Hidden(INPUT_SIZE, 768));
        net.add(new Activation()); // 768
        net.add(new Hidden(768, 384));
        net.add(new Activation()); // 384
        net.add(new Hidden(384, 256));
        net.add(new Activation()); // 256
        net.add(new Hidden(256, NUM_LABELS));
        net.add(new Activation()); // 10
//        net.add(new Softmax()); // XXX
        net.add(loss);

        final float learningRate = 0.005f;

        // shuffle images and labels randomly
        long seed = ThreadLocalRandom.current().nextLong();
        Statistics.shuffleColumnsInplace(IMAGES, seed);
        Statistics.shuffleColumnsInplace(EXPECT, seed);

        // train for 40 epochs
        for (int i = 0; i <= NUM_BATCHES * NUM_BATCHES_PER_EPOCH; ++i) {
            int startCol = getStartColumn(i);
            MatrixF input = IMAGES.selectConsecutiveColumns(startCol, startCol + BATCH_SIZE - 1);
            net.train(input, learningRate);
            if (i > 0 && (i % NUM_BATCHES_PER_EPOCH == 0)) {
                System.out.println(
                        "epoch " + epoch + "   : avg. accuracy: " + (epochAccuraciesSum / NUM_BATCHES_PER_EPOCH));
                epochAccuraciesSum = 0.0;
                ++epoch;
                // reshuffle before the next epoch
                seed = ThreadLocalRandom.current().nextLong();
                Statistics.shuffleColumnsInplace(IMAGES, seed);
                Statistics.shuffleColumnsInplace(EXPECT, seed);
            }
        }

        System.out.println("\nDone with training. Starting validation.");
        final MatrixF TEST_IMAGES = Statistics.zscoreInplace(MNIST.getTestSetImages());
        final MatrixF TEST_EXPECT = MNIST.getTestSetLabels();

        MatrixF predict = net.infer(TEST_IMAGES);
        double accuracy = CategorialAccuracy.computeAccuracy(predict, TEST_EXPECT);
        System.out.println("validation : avg. accuracy in validation: " + accuracy);
    }
}
