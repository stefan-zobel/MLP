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
 * A vision transformer on EMNIST Balanced, against the fully connected 0.8951 and the
 * convolutional 0.901862 on the same augmented stream.
 *
 * <p>The patch projection is a convolution whose stride equals its kernel, which is the same
 * operation as a linear projection of each tile and already emits its columns sample-major,
 * position-minor, so no reshaping layer is needed. The activation is a ReLU rather than the
 * usual GELU because both baselines use one, which leaves the attention as the only difference
 * between them and this.
 *
 * <p>Usage: {@code EMNIST_Transformer [seed] [epochs] [peak rate] [passes per epoch] [resume]}.
 * A nonzero last argument continues an interrupted run from the promoted checkpoint: the same
 * seed, the same epoch count and the same rate, so the schedule keeps its length, and only the
 * epochs that are left are trained.
 */
public class EMNIST_Transformer extends AbstractNetwork {

    public EMNIST_Transformer() {
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

    private static final int D_MODEL = 64;
    private static final int HEADS = 4;
    private static final int BLOCKS = 4;
    private static final int TILE = 4;
    private static final int MLP_WIDTH = 4 * D_MODEL;

    private static final int DEFAULT_PASSES_PER_EPOCH = 7;
    private static final int DEFAULT_EPOCHS = 2;
    // an order of magnitude under the 1e-2 the convolutional nets ran at, which is where a
    // transformer of this size is usually trained; the fourth argument exists to probe it
    private static final float DEFAULT_PEAK_RATE = 1e-3f;

    private static final MatrixF TEST_IMAGES = Statistics.rescaleInplace(EMNIST.getTestSetImages(), LOWER, UPPER);
    private static final MatrixF TEST_EXPECT = EMNIST.getTestSetLabels();

    private static double epochAccuraciesSum = 0.0;
    private static double epochLossesSum = 0.0;

    public static void main(String[] args) {
        long baseSeed = args.length > 0 ? Long.parseLong(args[0]) : new SecureRandom().nextLong();
        int epochs = args.length > 1 ? Integer.parseInt(args[1]) : DEFAULT_EPOCHS;
        float peakRate = args.length > 2 ? Float.parseFloat(args[2]) : DEFAULT_PEAK_RATE;
        int passesPerEpoch = args.length > 3 ? Integer.parseInt(args[3]) : DEFAULT_PASSES_PER_EPOCH;
        // continue a run that was stopped: the weights come from ./data/, the optimizer moments
        // and the step counter with them, and the epochs argument stays the whole plan so that
        // the schedule keeps the length it was built with
        boolean resume = args.length > 4 && Integer.parseInt(args[4]) != 0;

        SplittableRandom seeds = new SplittableRandom(baseSeed);
        EMNIST_Transformer net = new EMNIST_Transformer();
        SoftmaxCrossEntropyLoss loss = new SoftmaxCrossEntropyLoss();
        loss.registerAccuracyCallback(net::onAccuracyComputationCompleted);
        loss.registerLossCallback(net::onLossComputationCompleted);

        TransformerEncoder encoder = TransformerEncoder.builder()
                .image(IMAGE_SIZE, IMAGE_SIZE)
                .tile(TILE)
                .dModel(D_MODEL)
                .heads(HEADS)
                .blocks(BLOCKS)
                .activation(Relu::new)
                .names("vt_")
                .load(resume)
                .store(true)
                .seed(baseSeed)
                .build();
        int seqLen = encoder.sequenceLength();
        net.add(encoder);
        // the head belongs to the program, not to the encoder: what is classified, and into how
        // many classes, is not the encoder's business
        net.add(new Hidden(encoder.features(), NUM_LABELS, "vt_out", resume, true, seeds.nextLong()));
        // no activation here: SoftmaxCrossEntropyLoss wants raw logits
        net.add(loss);

        SplittableRandom passes = seeds.split();
        PassSource data = AugmentedSet.forEmnistTraining();

        int batchesPerPass = data.images().numColumns() / BATCH_SIZE;
        int batchesPerEpoch = passesPerEpoch * batchesPerPass;
        int totalSteps = batchesPerEpoch * epochs;
        long tokens = (long) D_MODEL * TILE * TILE + (long) D_MODEL * seqLen;
        long attention = (long) BLOCKS * 4 * D_MODEL * D_MODEL;
        long mlp = 2L * BLOCKS * D_MODEL * MLP_WIDTH;
        long head = (long) D_MODEL * NUM_LABELS;
        long all = tokens + attention + mlp + head;
        System.out.printf("seed=%d epochs=%d rate=%s passes=%d steps=%d seqLen=%d%n", baseSeed, epochs, peakRate,
                passesPerEpoch, totalSteps, seqLen);
        System.out.printf("weights=%d (tokens %d, attention %d, mlp %d, head %d)%n", all, tokens, attention, mlp,
                head);

        AbstractOptimizer adam = new Adam(
                LearningRateSchedule.warmupThenCosine(totalSteps / 20, peakRate, totalSteps, peakRate / 100.0f), 0.0f)
                        .persistAs("vt", true);
        net.optimizer(adam);

        double maxValidationAccuracy = 0.0;
        int bestEpoch = -1;
        int firstEpoch = 0;
        if (resume) {
            // after the registration, because the moments are sized from the parameters
            int done = adam.loadState();
            firstEpoch = done / batchesPerEpoch;
            bestEpoch = firstEpoch - 1;
            // what the keep-best rule has to beat is the promoted checkpoint, and after a resume
            // its score is only known by measuring it; starting from zero would store the next
            // epoch whatever it scored
            maxValidationAccuracy = net.validationAccuracy();
            System.out.printf("resuming at step %d, epoch %d, checkpoint accuracy %.6f%n", done, firstEpoch,
                    maxValidationAccuracy);
        }
        long t0 = System.nanoTime();

        for (int epoch = firstEpoch; epoch < epochs; ++epoch) {
            long e0 = System.nanoTime();
            for (int pass = 0; pass < passesPerEpoch; ++pass) {
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
                    + maxValidationAccuracy + "   : " + Arithmetic.round((System.nanoTime() - e0) / 1e9, 1) + "s");
            epochAccuraciesSum = 0.0;
            epochLossesSum = 0.0;
        }

        System.out.printf("%nseed=%d epochs=%d rate=%s passes=%d  best=%.6f (epoch %d)  %.1fs%n", baseSeed, epochs,
                peakRate, passesPerEpoch, maxValidationAccuracy, bestEpoch, (System.nanoTime() - t0) / 1e9);
    }

    /** Batched for the same reason {@link EMNIST_ConvNetwork1} batches: the whole test set at
     * once would hold every block's activations for 18 800 samples. */
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
