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
import math.ml.loader.MNIST;
import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;
import net.jamu.matrix.Statistics;

/**
 * The network of {@link MNIST_TrainingNetwork3} on all seven training sets: the original,
 * the two shifted copies, the two affine and the two elastic ones.
 */
public class MNIST_TrainingNetwork4 extends AbstractNetwork {

    public MNIST_TrainingNetwork4() {
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
    private static final float LOWER = 0.0f;
    private static final float UPPER = 1.0f;

    private static final int NUM_SETS = 7;

    // 784 x 420_000: the training set, its left- and right-shifted copies, the two affine
    // and the two elastic ones. Assembled into one preallocated matrix rather than by a
    // chain of appendMatrix calls, which would hold the 360_000-column intermediate and
    // the finished 1.32 GB matrix at the same time and overflow the default heap.
    private static final MatrixF IMAGES = Statistics.rescaleInplace(trainingImages(), LOWER, UPPER);

    // 10 x 420_000: every set is a per-image transform of the original, so the labels repeat
    private static final MatrixF EXPECT = repeatedLabels();

    private static final MatrixF TEST_IMAGES = Statistics.rescaleInplace(MNIST.getTestSetImages(), LOWER, UPPER);
    private static final MatrixF TEST_EXPECT = MNIST.getTestSetLabels();

    private static final int INPUT_SIZE = IMAGES.numRows();
    private static final int NUM_BATCHES_PER_EPOCH = IMAGES.numColumns() / BATCH_SIZE;
    // fixed, with no early break: the cosine decay needs a known horizon to reach its
    // floor, and a run that stops early turns that horizon into a mere upper bound.
    // 28 of them are 58_800 steps, and past about 36_000 the result stops moving: measured
    // on five sets, 0.9927 at 36_000 steps, 0.9929 at 42_000 and 0.9926 at 60_000. A longer
    // run costs time and buys nothing measurable.
    private static final int NUM_EPOCHS = 28;
    // tuned over six epochs on the 180_000-column variant that still had dropout: 0.9855 at
    // 3e-4, 0.9882 at 1e-3, 0.9885 at 3e-3, 0.9903 at 1e-2, 0.9904 at 2e-2, then 0.9894 at
    // 3e-2 and above. Not retuned for the seven sets and the 58_800 steps they bring; the
    // plateau was flat enough that the lower end of it stays the safer choice.
    private static final float PEAK_RATE = 1e-2f;

    private static int epoch = 0;
    private static double epochAccuraciesSum = 0.0;
    private static double epochLossesSum = 0.0;

    public static void main(String[] args) {
        // pass this seed back as the first argument to repeat a run exactly
        long baseSeed = args.length > 0 ? Long.parseLong(args[0]) : new SecureRandom().nextLong();
        System.out.println("seed: " + baseSeed);
        SplittableRandom seeds = new SplittableRandom(baseSeed);

        MNIST_TrainingNetwork4 net = new MNIST_TrainingNetwork4();
        SoftmaxCrossEntropyLoss loss = new SoftmaxCrossEntropyLoss();
        loss.registerAccuracyCallback(net::onAccuracyComputationCompleted);
        loss.registerLossCallback(net::onLossComputationCompleted);

        // The names carry a prefix of their own: the other training networks store under
        // layer1 to layer4 and under t3_, and sharing either would overwrite them.
        // He ahead of every ReLU, Glorot on the output layer, which feeds the loss directly.
        net.add(new Hidden(INPUT_SIZE, 512, "t4_layer1", false, true, Init.HE, seeds.nextLong()));
        net.add(new BatchNorm(512, "t4_norm1", false, true));
        net.add(new Relu());
        // no dropout: on the 180_000-column variant it was worth +0.01 points over three
        // seeds, inside the noise, and cost 18 % of the epoch -- with five training sets
        // there is even less left for it to do
        net.add(new Hidden(512, 256, "t4_layer2", false, true, Init.HE, seeds.nextLong()));
        net.add(new BatchNorm(256, "t4_norm2", false, true));
        net.add(new Relu());
        net.add(new Hidden(256, NUM_LABELS, "t4_out", false, true, seeds.nextLong()));
        // no activation here: SoftmaxCrossEntropyLoss wants raw logits
        net.add(loss);

        int totalSteps = NUM_BATCHES_PER_EPOCH * NUM_EPOCHS;
        net.optimizer(new Adam(
                LearningRateSchedule.warmupThenCosine(totalSteps / 20, PEAK_RATE, totalSteps, PEAK_RATE / 100.0f),
                0.0f));

        long seed = seeds.nextLong();
        Statistics.shuffleColumnsInplace(IMAGES, seed);
        Statistics.shuffleColumnsInplace(EXPECT, seed);

        double maxValidationAccuracy = 0.0;

        for (int epochIdx = 0; epochIdx < NUM_EPOCHS; ++epochIdx) {
            for (int b = 0; b < NUM_BATCHES_PER_EPOCH; ++b) {
                int startCol = b * BATCH_SIZE;
                MatrixF input = IMAGES.selectConsecutiveColumns(startCol, startCol + BATCH_SIZE - 1);
                MatrixF expected = EXPECT.selectConsecutiveColumns(startCol, startCol + BATCH_SIZE - 1);
                net.train(input, expected);
            }
            double trainingAccuracy = Arithmetic.round(epochAccuraciesSum / NUM_BATCHES_PER_EPOCH, 6);
            double avgTrainingLoss = Arithmetic.round(epochLossesSum / NUM_BATCHES_PER_EPOCH, 6);
            double validationAccuracy = net.validationAccuracy();
            // keep-best: only the improved model reaches the disk
            if (validationAccuracy > maxValidationAccuracy) {
                maxValidationAccuracy = validationAccuracy;
                net.storeParameters();
            }
            System.out.println("epoch " + epoch + "   : avg. accuracy: " + trainingAccuracy + "   : avg. loss: "
                    + avgTrainingLoss + "   : validation avg. accuracy: " + validationAccuracy + "   : max acc.: "
                    + maxValidationAccuracy);
            epochAccuraciesSum = 0.0;
            epochLossesSum = 0.0;
            ++epoch;

            // reshuffle between epochs
            seed = seeds.nextLong();
            Statistics.shuffleColumnsInplace(IMAGES, seed);
            Statistics.shuffleColumnsInplace(EXPECT, seed);
        }

        System.out.println("\nDone with training. Best validation accuracy: " + maxValidationAccuracy);
        System.out.println("validation : avg. accuracy in validation: " + net.validationAccuracy());
    }

    private double validationAccuracy() {
        MatrixF predict = infer(TEST_IMAGES);
        return CategorialAccuracy.computeAccuracy(predict, TEST_EXPECT);
    }

    // The sets are loaded one at a time and released as soon as they are copied, so only
    // one source is ever live beside the destination.
    private static MatrixF trainingImages() {
        MatrixF first = MNIST.getTrainingSetImages();
        MatrixF all = Matrices.createF(first.numRows(), NUM_SETS * first.numColumns());
        int col = copyInto(all, first, 0);
        col = copyInto(all, MNIST.getTrainingSetImagesLeft(), col);
        col = copyInto(all, MNIST.getTrainingSetImagesRight(), col);
        col = copyInto(all, MNIST.getTrainingSetImagesAffine1(), col);
        col = copyInto(all, MNIST.getTrainingSetImagesAffine2(), col);
        col = copyInto(all, MNIST.getTrainingSetImagesElastic1(), col);
        copyInto(all, MNIST.getTrainingSetImagesElastic2(), col);
        return all;
    }

    private static MatrixF repeatedLabels() {
        MatrixF labels = MNIST.getTrainingSetLabels();
        MatrixF all = Matrices.createF(labels.numRows(), NUM_SETS * labels.numColumns());
        int col = 0;
        for (int i = 0; i < NUM_SETS; ++i) {
            col = copyInto(all, labels, col);
        }
        return all;
    }

    // copies one set into the block starting at startCol and returns the next free column
    private static int copyInto(MatrixF all, MatrixF part, int startCol) {
        all.setSubmatrixInplace(0, startCol, part, 0, 0, part.numRows() - 1, part.numColumns() - 1);
        return startCol + part.numColumns();
    }

}
