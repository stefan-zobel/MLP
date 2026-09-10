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

import java.security.SecureRandom;
import java.util.SplittableRandom;

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
        // pass this seed back as the first argument to repeat a run exactly
        long baseSeed = args.length > 0 ? Long.parseLong(args[0]) : new SecureRandom().nextLong();
        System.out.println("seed: " + baseSeed);
        SplittableRandom seeds = new SplittableRandom(baseSeed);

        MNIST_TrainingNetwork net = new MNIST_TrainingNetwork();
//        CrossEntropyLoss loss = new CrossEntropyLoss(); // XXX
        SoftmaxCrossEntropyLoss loss = new SoftmaxCrossEntropyLoss();
        loss.registerAccuracyCallback(net::onAccuracyComputationCompleted);
        loss.registerLossCallback(net::onLossComputationCompleted);

        // He ahead of every ReLU, Glorot on the output layer, which feeds the loss directly
        net.add(new Hidden(INPUT_SIZE, 768, "layer1", false, true, Init.HE, seeds.nextLong()));
        net.add(new Relu()); // 768
        net.add(new Hidden(768, 384, "layer2", false, true, Init.HE, seeds.nextLong()));
        net.add(new Relu()); // 384
        net.add(new Hidden(384, 256, "layer3", false, true, Init.HE, seeds.nextLong()));
        net.add(new Relu()); // 256
        net.add(new Hidden(256, NUM_LABELS, "layer4", false, true, seeds.nextLong()));
        // no activation here: SoftmaxCrossEntropyLoss wants raw logits
        net.add(loss);

        net.optimizer(new Sgd(0.001f)); // XXX ?

        // shuffle images and labels randomly, both with the same seed so they stay aligned
        long seed = seeds.nextLong();
        Statistics.shuffleColumnsInplace(IMAGES, seed);
        Statistics.shuffleColumnsInplace(EXPECT, seed);

        double maxValidationAccuracy = 0.0;

        // train for up to NUM_BATCHES epochs
        for (int epochIdx = 0; epochIdx < NUM_BATCHES; ++epochIdx) {
            for (int b = 0; b < NUM_BATCHES_PER_EPOCH; ++b) {
                int startCol = b * BATCH_SIZE;
                MatrixF input = IMAGES.selectConsecutiveColumns(startCol, startCol + BATCH_SIZE - 1);
                MatrixF expected = EXPECT.selectConsecutiveColumns(startCol, startCol + BATCH_SIZE - 1);
                net.train(input, expected);
            }
            double trainingAccuracy = Arithmetic.round(epochAccuraciesSum / NUM_BATCHES_PER_EPOCH, 6);
            double validationAccuracy = net.validationAccuracy();
            System.out.println("epoch " + epoch + "   : avg. accuracy: " + trainingAccuracy + "   : avg. loss: "
                    + Arithmetic.round(epochLossesSum / NUM_BATCHES_PER_EPOCH, 6)
                    + "   : validation avg. accuracy: " + validationAccuracy);
            epochAccuraciesSum = 0.0;
            epochLossesSum = 0.0;
            ++epoch;
            // keep-best: only the improved model reaches the disk
            if (validationAccuracy > maxValidationAccuracy) {
                maxValidationAccuracy = validationAccuracy;
                net.storeParameters();
            }
            if (epoch > 5 && validationAccuracy < trainingAccuracy - 0.05) {
                System.out.println("potential overfitting. BREAK.");
                break;
            }
            // reshuffle between epochs
            seed = seeds.nextLong();
            Statistics.shuffleColumnsInplace(IMAGES, seed);
            Statistics.shuffleColumnsInplace(EXPECT, seed);
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
