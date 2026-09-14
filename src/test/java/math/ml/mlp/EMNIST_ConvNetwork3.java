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
 * {@link EMNIST_ConvNetwork2} with the pooling ahead of the activation, which computes the
 * same thing: a ReLU is monotone, so the largest activation of a window is the activation of
 * its largest element, and both agree backward because the derivative at zero is zero.
 *
 * <p>The activation then runs on a quarter of the elements in the first block and a fifth in
 * the last, which is worth about a tenth of the epoch.
 */
public class EMNIST_ConvNetwork3 extends AbstractNetwork {

    public EMNIST_ConvNetwork3() {
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
    private static final int DEFAULT_EPOCHS = 2;
    // the peak the three-arm probe found for EMNIST_ConvNetwork1; carried over here and
    // not re-probed, which is a caveat on the absolute level rather than on the comparison
    private static final float DEFAULT_PEAK_RATE = 1e-2f;

    private static final MatrixF TEST_IMAGES = Statistics.rescaleInplace(EMNIST.getTestSetImages(), LOWER, UPPER);
    private static final MatrixF TEST_EXPECT = EMNIST.getTestSetLabels();

    private static double epochAccuraciesSum = 0.0;
    private static double epochLossesSum = 0.0;

    public static void main(String[] args) {
        long baseSeed = args.length > 0 ? Long.parseLong(args[0]) : new SecureRandom().nextLong();
        int epochs = args.length > 1 ? Integer.parseInt(args[1]) : DEFAULT_EPOCHS;
        float peakRate = args.length > 2 ? Float.parseFloat(args[2]) : DEFAULT_PEAK_RATE;
        // continue a run that was stopped: the bundle comes from ./data/, and the epochs
        // argument stays the whole plan so the schedule keeps the length it was built with
        boolean resume = args.length > 3 && Integer.parseInt(args[3]) != 0;

        SplittableRandom seeds = new SplittableRandom(baseSeed);
        EMNIST_ConvNetwork3 net = new EMNIST_ConvNetwork3();
        SoftmaxCrossEntropyLoss loss = new SoftmaxCrossEntropyLoss();
        loss.registerAccuracyCallback(net::onAccuracyComputationCompleted);
        loss.registerLossCallback(net::onLossComputationCompleted);

        String prefix = "c3_";
        net.add(new Unflatten(1, IMAGE_SIZE, IMAGE_SIZE));
        net.add(new Conv2D(1, 32, 28, 28, 3, 1, 1, prefix + "conv1", Init.HE, seeds.nextLong()));
        net.add(new BatchNorm(32, prefix + "norm1"));
        net.add(new MaxPool2D(32, 28, 28, 2));
        net.add(new Relu());
        net.add(new Conv2D(32, 64, 14, 14, 3, 1, 1, prefix + "conv2", Init.HE, seeds.nextLong()));
        net.add(new BatchNorm(64, prefix + "norm2"));
        net.add(new MaxPool2D(64, 14, 14, 2));
        net.add(new Relu());
        net.add(new Conv2D(64, 128, 7, 7, 3, 1, 1, prefix + "conv3", Init.HE, seeds.nextLong()));
        net.add(new BatchNorm(128, prefix + "norm3"));
        // 7 does not halve: (7 - 2) / 2 + 1 leaves 3x3 and drops the last row and column
        net.add(new MaxPool2D(128, 7, 7, 2));
        net.add(new Relu());
        net.add(new Flatten(128, 3, 3));
        net.add(new Hidden(128 * 3 * 3, 128, prefix + "dense", Init.HE, seeds.nextLong()));
        net.add(new BatchNorm(128, prefix + "norm4"));
        net.add(new Relu());
        net.add(new Hidden(128, NUM_LABELS, prefix + "out", seeds.nextLong()));
        // no activation here: SoftmaxCrossEntropyLoss wants raw logits
        net.add(loss);

        SplittableRandom passes = seeds.split();
        PassSource data = AugmentedSet.forEmnistTraining();

        int batchesPerPass = data.images().numColumns() / BATCH_SIZE;
        int batchesPerEpoch = PASSES_PER_EPOCH * batchesPerPass;
        int totalSteps = batchesPerEpoch * epochs;
        long conv = 32L * 9 + 64L * 32 * 9 + 128L * 64 * 9;
        long dense = 128L * 3 * 3 * 128 + 128L * NUM_LABELS;
        System.out.printf("seed=%d epochs=%d rate=%s steps=%d weights=%d (conv %d, %.1f%%)%n", baseSeed, epochs,
                peakRate, totalSteps, conv + dense, conv, 100.0 * conv / (conv + dense));

        net.optimizer(new Adam(
                LearningRateSchedule.warmupThenCosine(totalSteps / 20, peakRate, totalSteps, peakRate / 100.0f),
                0.0f));

        double maxValidationAccuracy = 0.0;
        int bestEpoch = -1;
        int firstEpoch = 0;
        if (resume) {
            firstEpoch = net.resume("c3", batchesPerEpoch);
            // put the data stream back where the uninterrupted run would have been: the pass
            // is a pure function of the generator, so replaying its draws is enough
            for (int p = 0; p < firstEpoch * PASSES_PER_EPOCH; ++p) {
                data.skip(passes);
            }
            // what keep-best has to beat is the bundle that was just loaded, and its score is
            // only known by measuring it; from zero the next epoch would store whatever it scored
            maxValidationAccuracy = net.validationAccuracy();
            bestEpoch = firstEpoch - 1;
            System.out.printf("resuming at epoch %d, checkpoint accuracy %.6f%n", firstEpoch,
                    maxValidationAccuracy);
        }
        long t0 = System.nanoTime();

        for (int epoch = firstEpoch; epoch < epochs; ++epoch) {
            long e0 = System.nanoTime();
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
                net.storeParameters("c3");
            }
            System.out.println("epoch " + epoch + "   : avg. accuracy: " + trainingAccuracy + "   : avg. loss: "
                    + avgTrainingLoss + "   : validation avg. accuracy: " + validationAccuracy + "   : max acc.: "
                    + maxValidationAccuracy + "   : " + Arithmetic.round((System.nanoTime() - e0) / 1e9, 1) + "s");
            epochAccuraciesSum = 0.0;
            epochLossesSum = 0.0;
        }

        System.out.printf("%nseed=%d epochs=%d rate=%s  best=%.4f (epoch %d)  %.1fs%n", baseSeed, epochs, peakRate,
                maxValidationAccuracy, bestEpoch, (System.nanoTime() - t0) / 1e9);
    }

    /** Batched for the same reason {@link EMNIST_ConvNetwork1} batches: the whole test set at
     * once would cost more than a gigabyte in the first convolution alone. */
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
            sum += CategorialAccuracy.computeAccuracy(infer(TEST_IMAGES.selectConsecutiveColumns(from, to)),
                    TEST_EXPECT.selectConsecutiveColumns(from, to));
        }
        return Arithmetic.round(sum / batches, 6);
    }
}
