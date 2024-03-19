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

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;
import net.jamu.matrix.Statistics;

public class RandomByRoteNetwork extends AbstractNetwork {

    public RandomByRoteNetwork() {
    }

    public void onAccuracyComputationCompleted(double accuracy) {
        System.out.println("Accuracy: " + accuracy);
        if (accuracy == 1.0) {
            stop = true;
            System.out.println("Stopped after " + batchCount + " batches");
        }
    }

    public MatrixF getExpectedBatchResults(int batchNumber) {
        return EXPECT;
    }

    private static boolean stop = false;
    private static final int INPUT_SIZE = 28 * 28; // 784
    private static final int NUM_LABELS = 10;
    private static final int BATCH_SIZE = 200;
    // 10 different labels, randomly assigned
    static final MatrixF EXPECT = Matrices.createF(NUM_LABELS, BATCH_SIZE);
    static {
        ThreadLocalRandom rng = ThreadLocalRandom.current();
        for (int i = 0; i < BATCH_SIZE; ++i) {
            // attach a random label in [0..9]
            EXPECT.set(rng.nextInt(NUM_LABELS), i, 1.0f);
        }
    }

    public static void main(String[] args) {
        RandomByRoteNetwork net = new RandomByRoteNetwork();
        CrossEntropyLoss loss = new CrossEntropyLoss();
        loss.registerLossCallback(net::onLossComputationCompleted);
        loss.registerAccuracyCallback(net::onAccuracyComputationCompleted);
        loss.registerBatchExpectedValuesProvider(net::getExpectedBatchResults);

        net.add(new Hidden(INPUT_SIZE, 1024, "1"));
        net.add(new Gelu()); // 1024
        net.add(new Hidden(1024, 1024, "2"));
        net.add(new Gelu()); // 1024
        net.add(new Hidden(1024, 1024, "3"));
        net.add(new Gelu()); // 1024
        net.add(new Hidden(1024, 1024, "4"));
        net.add(new Gelu()); // 1024
        net.add(new Hidden(1024, 512, "5"));
        net.add(new Gelu()); // 512
        net.add(new Hidden(512, 256, "6"));
        net.add(new Gelu()); // 256
        net.add(new Hidden(256, NUM_LABELS, "7"));
        net.add(new Gelu()); // 10
        net.add(new Softmax());
        net.add(loss);

        MatrixF input = Matrices.randomUniformF(INPUT_SIZE, BATCH_SIZE, -1.0f, 1.0f);
        input = Statistics.zscoreInplace(input);

        final float learningRate = 0.008f;

        for (int i = 0; i < 10_000 && !stop; ++i) {
            net.train(input, learningRate);
        }

        System.out.println("\nDone.");
    }
}
