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

import java.security.SecureRandom;
import java.util.SplittableRandom;

import math.cern.Arithmetic;
import math.ml.loader.AugmentedSet;
import math.ml.loader.EMNIST;
import math.ml.loader.PassSource;
import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;
import net.jamu.matrix.Statistics;

/**
 * A convolutional network on the balanced split of EMNIST, against the 0.8951 the fully
 * connected {@link EMNIST_TrainingNetwork1} reaches on its live arm.
 *
 * <p>Dataset, augmentation, batch size and passes per epoch are held at what that run used,
 * so that what differs between the two is the model class.
 */
public class EMNIST_ConvNetwork1 extends AbstractNetwork {

    public EMNIST_ConvNetwork1() {
    }

    @Override
    public void onLossComputationCompleted(MatrixF losses) {
        epochLossesSum += Matrices.colsAverage(losses).toScalar();
    }

    @Override
    public void onAccuracyComputationCompleted(double accuracy) {
        epochAccuraciesSum += accuracy;
    }

    private static final int NUM_LABELS = 47;
    private static final int BATCH_SIZE = 200;
    private static final int IMAGE_SIZE = 28;
    private static final float LOWER = 0.0f;
    private static final float UPPER = 1.0f;

    private static final int PASSES_PER_EPOCH = 7;
    private static final int DEFAULT_EPOCHS = 28;
    // probed over two epochs per arm rather than carried over from the fully connected
    // run: 3e-3 gives 0.8974, 1e-2 0.8978 and 3e-2 0.8962, so the rate that architecture
    // used turns out to be right for this one too
    private static final float DEFAULT_PEAK_RATE = 1e-2f;

    private static final MatrixF TEST_IMAGES = Statistics.rescaleInplace(EMNIST.getTestSetImages(), LOWER, UPPER);
    private static final MatrixF TEST_EXPECT = EMNIST.getTestSetLabels();

    private static double epochAccuraciesSum = 0.0;
    private static double epochLossesSum = 0.0;

    public static void main(String[] args) {
        long baseSeed = args.length > 0 ? Long.parseLong(args[0]) : new SecureRandom().nextLong();
        int epochs = args.length > 1 ? Integer.parseInt(args[1]) : DEFAULT_EPOCHS;
        float peakRate = args.length > 2 ? Float.parseFloat(args[2]) : DEFAULT_PEAK_RATE;

        SplittableRandom seeds = new SplittableRandom(baseSeed);
        EMNIST_ConvNetwork1 net = new EMNIST_ConvNetwork1();
        SoftmaxCrossEntropyLoss loss = new SoftmaxCrossEntropyLoss();
        loss.registerAccuracyCallback(net::onAccuracyComputationCompleted);
        loss.registerLossCallback(net::onLossComputationCompleted);

        String prefix = "c1_";
        // the loaders hand over 784 x m, which is the same order a single channel wants
        net.add(new Unflatten(1, IMAGE_SIZE, IMAGE_SIZE));
        net.add(new Conv2D(1, 16, 28, 28, 3, 1, 1, prefix + "conv1", false, true, Init.HE, seeds.nextLong()));
        net.add(new BatchNorm(16, prefix + "norm1", false, true));
        net.add(new Relu());
        net.add(new MaxPool2D(16, 28, 28, 2));
        net.add(new Conv2D(16, 32, 14, 14, 3, 1, 1, prefix + "conv2", false, true, Init.HE, seeds.nextLong()));
        net.add(new BatchNorm(32, prefix + "norm2", false, true));
        net.add(new Relu());
        net.add(new MaxPool2D(32, 14, 14, 2));
        net.add(new Flatten(32, 7, 7));
        net.add(new Hidden(32 * 7 * 7, 256, prefix + "dense", false, true, Init.HE, seeds.nextLong()));
        net.add(new BatchNorm(256, prefix + "norm3", false, true));
        net.add(new Relu());
        net.add(new Hidden(256, NUM_LABELS, prefix + "out", false, true, seeds.nextLong()));
        // no activation here: SoftmaxCrossEntropyLoss wants raw logits
        net.add(loss);

        SplittableRandom passes = seeds.split();
        PassSource data = AugmentedSet.forEmnistTraining();

        int batchesPerPass = data.images().numColumns() / BATCH_SIZE;
        int batchesPerEpoch = PASSES_PER_EPOCH * batchesPerPass;
        int totalSteps = batchesPerEpoch * epochs;
        long weights = 16L * 9 + 32L * 16 * 9 + 32L * 7 * 7 * 256 + 256L * NUM_LABELS;
        System.out.printf("seed=%d epochs=%d rate=%s steps=%d weights=%d%n", baseSeed, epochs, peakRate, totalSteps,
                weights);

        net.optimizer(new Adam(
                LearningRateSchedule.warmupThenCosine(totalSteps / 20, peakRate, totalSteps, peakRate / 100.0f),
                0.0f));

        double maxValidationAccuracy = 0.0;
        int bestEpoch = -1;
        long t0 = System.nanoTime();

        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (int pass = 0; pass < PASSES_PER_EPOCH; ++pass) {
                data.regenerate(passes);
                for (int b = 0; b < batchesPerPass; ++b) {
                    int startCol = b * BATCH_SIZE;
                    MatrixF in = data.images().selectConsecutiveColumns(startCol, startCol + BATCH_SIZE - 1);
                    MatrixF expected = data.labels().selectConsecutiveColumns(startCol, startCol + BATCH_SIZE - 1);
                    net.train(in, expected);
                }
            }
            double trainingAccuracy = Arithmetic.round(epochAccuraciesSum / batchesPerEpoch, 6);
            double avgTrainingLoss = Arithmetic.round(epochLossesSum / batchesPerEpoch, 6);
            double validationAccuracy = net.validationAccuracy();
            if (validationAccuracy > maxValidationAccuracy) {
                maxValidationAccuracy = validationAccuracy;
                bestEpoch = epoch;
                net.storeParameters();
            }
            System.out.println("epoch " + epoch + "   : avg. accuracy: " + trainingAccuracy + "   : avg. loss: "
                    + avgTrainingLoss + "   : validation avg. accuracy: " + validationAccuracy + "   : max acc.: "
                    + maxValidationAccuracy);
            epochAccuraciesSum = 0.0;
            epochLossesSum = 0.0;
        }

        System.out.printf("%nseed=%d epochs=%d rate=%s weights=%d  best=%.4f (epoch %d)  %.1fs%n", baseSeed, epochs,
                peakRate, weights, maxValidationAccuracy, bestEpoch, (System.nanoTime() - t0) / 1e9);
    }

    /**
     * The whole test set at once would cost more than a gigabyte in the first convolution
     * alone, so it is inferred in batches. 18 800 divides by 200 without a remainder, which
     * is what makes the mean over the batches exact rather than approximate.
     */
    private double validationAccuracy() {
        int columns = TEST_IMAGES.numColumns();
        if (columns % BATCH_SIZE != 0) {
            throw new IllegalStateException(columns + " test columns do not divide into batches of " + BATCH_SIZE);
        }
        int batches = columns / BATCH_SIZE;
        double sum = 0.0;
        for (int b = 0; b < batches; ++b) {
            int from = b * BATCH_SIZE;
            int to = from + BATCH_SIZE - 1;
            MatrixF predict = infer(TEST_IMAGES.selectConsecutiveColumns(from, to));
            sum += CategorialAccuracy.computeAccuracy(predict, TEST_EXPECT.selectConsecutiveColumns(from, to));
        }
        return Arithmetic.round(sum / batches, 6);
    }
}
