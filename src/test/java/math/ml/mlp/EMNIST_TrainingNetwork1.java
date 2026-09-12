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
import math.ml.loader.FrozenAugmentedSet;
import math.ml.loader.PassSource;
import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;
import net.jamu.matrix.Statistics;

/**
 * The network of {@link MNIST_TrainingNetwork5} on the balanced split of EMNIST, trained from
 * one of three sources so that what per-pass distortion is worth can be read off directly.
 */
public class EMNIST_TrainingNetwork1 extends AbstractNetwork {

    public EMNIST_TrainingNetwork1() {
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
    private static final float LOWER = 0.0f;
    private static final float UPPER = 1.0f;

    private static final int PASSES_PER_EPOCH = 7;
    // the same budget in all three arms; what is being compared is where a pass comes from,
    // so anything else that differed between them would spoil the reading
    private static final int NUM_EPOCHS = 28;
    // carried over from MNIST rather than retuned. All three arms run at the same rate, so
    // only the absolute level could be slightly off, not the comparison between them.
    private static final float PEAK_RATE = 1e-2f;

    private static final MatrixF TEST_IMAGES = Statistics.rescaleInplace(EMNIST.getTestSetImages(), LOWER, UPPER);
    private static final MatrixF TEST_EXPECT = EMNIST.getTestSetLabels();

    private static int epoch = 0;
    private static double epochAccuraciesSum = 0.0;
    private static double epochLossesSum = 0.0;

    public static void main(String[] args) {
        // pass the seed back as the first argument to repeat a run exactly
        long baseSeed = args.length > 0 ? Long.parseLong(args[0]) : new SecureRandom().nextLong();
        String arm = args.length > 1 ? args[1] : "live";
        System.out.println("seed: " + baseSeed + "   arm: " + arm);

        SplittableRandom seeds = new SplittableRandom(baseSeed);
        EMNIST_TrainingNetwork1 net = new EMNIST_TrainingNetwork1();
        SoftmaxCrossEntropyLoss loss = new SoftmaxCrossEntropyLoss();
        loss.registerAccuracyCallback(net::onAccuracyComputationCompleted);
        loss.registerLossCallback(net::onLossComputationCompleted);

        // The three arms share their weights at a given seed because the layers draw first
        // and the data source only afterwards, from a generator split off behind them.
        String prefix = "e1_" + arm + "_";
        net.add(new Hidden(784, 512, prefix + "layer1", false, true, Init.HE, seeds.nextLong()));
        net.add(new BatchNorm(512, prefix + "norm1", false, true));
        net.add(new Relu());
        net.add(new Hidden(512, 256, prefix + "layer2", false, true, Init.HE, seeds.nextLong()));
        net.add(new BatchNorm(256, prefix + "norm2", false, true));
        net.add(new Relu());
        net.add(new Hidden(256, NUM_LABELS, prefix + "out", false, true, seeds.nextLong()));
        // no activation here: SoftmaxCrossEntropyLoss wants raw logits
        net.add(loss);

        SplittableRandom passes = seeds.split();
        PassSource data = source(arm, passes);

        int batchesPerPass = data.images().numColumns() / BATCH_SIZE;
        int batchesPerEpoch = PASSES_PER_EPOCH * batchesPerPass;
        int totalSteps = batchesPerEpoch * NUM_EPOCHS;
        System.out.println("columns: " + data.images().numColumns() + "   steps: " + totalSteps);

        net.optimizer(new Adam(
                LearningRateSchedule.warmupThenCosine(totalSteps / 20, PEAK_RATE, totalSteps, PEAK_RATE / 100.0f),
                0.0f));

        double maxValidationAccuracy = 0.0;

        for (int epochIdx = 0; epochIdx < NUM_EPOCHS; ++epochIdx) {
            for (int pass = 0; pass < PASSES_PER_EPOCH; ++pass) {
                data.regenerate(passes);
                for (int b = 0; b < batchesPerPass; ++b) {
                    int startCol = b * BATCH_SIZE;
                    MatrixF input = data.images().selectConsecutiveColumns(startCol, startCol + BATCH_SIZE - 1);
                    MatrixF expected = data.labels().selectConsecutiveColumns(startCol, startCol + BATCH_SIZE - 1);
                    net.train(input, expected);
                }
            }
            double trainingAccuracy = Arithmetic.round(epochAccuraciesSum / batchesPerEpoch, 6);
            double avgTrainingLoss = Arithmetic.round(epochLossesSum / batchesPerEpoch, 6);
            double validationAccuracy = net.validationAccuracy();
            // keep-best: only the improved model reaches the disk
            if (validationAccuracy > maxValidationAccuracy) {
                maxValidationAccuracy = validationAccuracy;
                net.storeParameters();
            }
            System.out.println(arm + " epoch " + epoch + "   : avg. accuracy: " + trainingAccuracy + "   : avg. loss: "
                    + avgTrainingLoss + "   : validation avg. accuracy: " + validationAccuracy + "   : max acc.: "
                    + maxValidationAccuracy);
            epochAccuraciesSum = 0.0;
            epochLossesSum = 0.0;
            ++epoch;
        }

        System.out.println("\nDone with training (" + arm + "). Best validation accuracy: " + maxValidationAccuracy);
        System.out.println("validation : avg. accuracy in validation: " + net.validationAccuracy());
    }

    // plain and live differ in whether a distortion is drawn at all, frozen and live in
    // whether the drawn ones are kept; between the three of them that separates what the
    // distortions are worth from what keeping them fresh is worth
    private static PassSource source(String arm, SplittableRandom rnd) {
        switch (arm) {
        case "plain":
            return AugmentedSet.forEmnistTraining(1, 0, 0);
        case "frozen":
            return FrozenAugmentedSet.of(AugmentedSet.forEmnistTraining(), PASSES_PER_EPOCH, rnd);
        case "live":
            return AugmentedSet.forEmnistTraining();
        default:
            throw new IllegalArgumentException("arm must be plain, frozen or live, not " + arm);
        }
    }

    private double validationAccuracy() {
        MatrixF predict = infer(TEST_IMAGES);
        return CategorialAccuracy.computeAccuracy(predict, TEST_EXPECT);
    }
}
