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

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

// The rest of the suite compares a run against another run of the same build, which
// catches nondeterminism but not a change in the numbers. These values were recorded
// once and must not move: they are what says an optimization was only an optimization.
//
// Deliberately free of Hidden, and therefore of sgemm: MKL varies its reduction order
// unless MKL_CBWR is set, which would make a golden flake.
class GoldenValuesTest {

    // JUnit compares floats by their bits, so this also pins the sign of every zero
    private static final float[] RELU_FORWARD = { 0.9278126f, 1.4883463f, 0.0f, 1.1772192f, 0.0f, 0.0f, 1.7814262f,
            0.0f, 0.100913525f, 0.24111843f, 1.725327f, 0.0f };
    private static final float[] RELU_BACKWARD = { 0.4644438f, -0.87002015f, 0.0f, -0.21795785f, 0.0f, -0.0f,
            0.19703293f, 0.0f, -0.170807f, 0.5453292f, 0.8252708f, 0.0f };

    private static final float[] GELU_FORWARD = { 0.7636923f, 1.3864276f, -0.15399379f, 1.0362697f, -0.1594734f,
            -0.1700288f, 1.7146833f, -0.11666041f, 0.05451249f, 0.1435291f, 1.6523418f, -0.16358097f };
    private static final float[] GELU_BACKWARD = { 0.49397084f, -0.9814203f, 0.09831916f, -0.24303813f, 0.032655396f,
            0.002372418f, 0.21844873f, -0.007960587f, -0.09910962f, 0.37556133f, 0.91916883f, 0.025021153f };

    private static final float[] SIGMOID_FORWARD = { 0.7166313f, 0.81582993f, 0.37805596f, 0.76444745f, 0.36739308f,
            0.31866154f, 0.8558729f, 0.20199989f, 0.525207f, 0.5599893f, 0.8488137f, 0.3572921f };
    private static final float[] SIGMOID_BACKWARD = { 0.094315015f, -0.1307218f, 0.17254515f, -0.039247133f,
            0.071236946f, -0.16133346f, 0.02430489f, 0.00998053f, -0.042593222f, 0.13436982f, 0.105906166f,
            0.070422344f };

    private static final float[] MASK_SEED_SEVEN = { 1.6666666f, 1.6666666f, 0.0f, 1.6666666f, 1.6666666f, 1.6666666f,
            1.6666666f, 0.0f, 1.6666666f, 0.0f, 1.6666666f, 1.6666666f };

    private static final float[] CHAIN_LOSSES = { 4.7095165f, 4.754839f, 5.247073f, 5.1949325f };

    // BatchNorm divides by the column count and LayerNorm by the feature count, both by
    // multiplying with the float reciprocal. Seven on the relevant axis makes that inexact,
    // so the fixture discriminates a plain division; a power of two would hide it.
    private static final int BN_D = 6;
    private static final int BN_M = 7;
    private static final int LN_D = 7;
    private static final int LN_M = 5;

    private static final float[] BN_FORWARD = { 0.23218127f, -0.5216418f, -0.7016284f, 1.1088233f, 0.4537768f,
            0.11575112f, -0.2394418f, -0.8906371f, 0.7931501f, -1.1812067f, 1.0276277f, 0.5367174f, -0.98699063f,
            1.3287905f, -1.4754467f, -1.2252694f, -0.3210402f, -0.35550097f, 1.4098165f, 0.108298525f, 0.4438814f,
            -0.33336607f, -1.9011518f, -1.2653513f, -1.4121932f, -0.9923884f, 0.20364757f, 1.527758f, 0.18590665f,
            -1.5400537f, 1.2676712f, -0.6406903f, -0.9777544f, -0.3431806f, 1.0226656f, 1.3490223f, -0.5726643f,
            1.748498f, 1.6915448f, 0.43465155f, -0.5392982f, 1.1975044f };
    private static final float[] BN_BACKWARD = { 0.02962643f, -0.03347273f, 0.13670662f, -0.7777203f, 0.34532058f,
            0.34341365f, -0.33515185f, -0.0835277f, 0.05324903f, 0.14702606f, 0.08751233f, -0.8044784f, 0.4744134f,
            -0.38699943f, -0.3169164f, -0.47114468f, -0.65594673f, -0.3078771f, 0.22185344f, -0.60901546f,
            -0.58249927f, 0.12799546f, 0.07196234f, -0.15526238f, 0.39555877f, 1.0454056f, -0.008113461f,
            0.107950404f, 0.25422657f, 0.42119566f, 0.095994115f, -0.5891343f, 0.5123161f, 0.051244617f, -0.270964f,
            0.8735156f, -0.88229454f, 0.6567441f, 0.20525713f, 0.81464833f, 0.16788898f, -0.3705069f };
    // gamma, beta, runningMean and runningVar end to end, after two training steps
    private static final float[] BN_PARAMS = { 0.9984733f, 1.0413735f, 1.0499685f, 0.98249245f, 0.9048718f, 1.0914133f,
            -0.08617743f, 0.040065594f, -0.006458688f, -0.003368722f, -0.020432321f, 0.010882504f, 0.13489796f,
            0.07723726f, 0.044126943f, -0.0948455f, 0.104978964f, 0.11567406f, 0.9264551f, 0.9536139f, 0.988799f,
            0.9333475f, 1.0898134f, 0.9296015f };
    private static final float[] BN_INFER = { 0.7341161f, -0.10082396f, -0.5057583f, 0.5034192f, 0.87992895f, 0.6638441f,
            0.35080194f, -0.43599823f, 0.9880256f, -1.3896075f, 1.5136983f, 1.0253962f, -0.25677222f, 1.5800033f,
            -1.2790616f, -1.4260315f, 0.024209479f, 0.25910342f, 1.6912451f, 0.47137788f, 0.6389893f, -0.6887498f,
            -1.7208891f, -0.52233297f, -0.6023578f, -0.52842337f, 0.39891535f, 0.84972686f, 0.5840891f, -0.7582645f,
            1.5757158f, -0.20896086f, -0.7817006f, -0.6968629f, 1.508218f, 1.7230545f, 0.07997369f, 1.9612415f,
            1.8858227f, -0.053876925f, -0.2168378f, 1.5929214f };

    private static final float[] LN_FORWARD = { 0.89182895f, 0.79319006f, 1.4182012f, -1.7392229f, -0.16019024f,
            -0.73114204f, -0.48290178f, 1.4210542f, 0.52418613f, -1.2589258f, -0.75621927f, 1.1722263f, -1.066696f,
            -0.22274241f, 0.09037234f, 1.4236279f, -1.5336106f, -0.6392651f, 1.3022199f, -0.15355003f, -0.69031155f,
            -0.995184f, -0.34738564f, 0.18644887f, -1.4661406f, 0.5686733f, 1.8349684f, 0.16454898f, -0.24132857f,
            0.99834675f, 1.6362009f, -0.95980245f, 0.49717158f, -0.4592139f, -1.4651653f };
    private static final float[] LN_BACKWARD = { 0.3148712f, 0.6542052f, -0.53383195f, 0.5262487f, 0.49592698f,
            -0.6127079f, -0.84471214f, 0.13242821f, 0.6974964f, 0.18357442f, -0.058044948f, -0.2060905f, 0.39614478f,
            -1.1455082f, -0.020577658f, -0.5959149f, 0.08516309f, -0.22150734f, 0.7766367f, -0.3610542f, 0.33725396f,
            -0.2962748f, 0.27328452f, -0.7734766f, 0.10323426f, 0.1357979f, -0.06215901f, 0.6195937f, 0.13644409f,
            1.2267615f, -1.7213614f, -1.5726054f, 0.67963946f, 1.3221854f, -0.07106408f };
    private static final float[] LN_PARAMS = { 0.98691714f, 0.9792012f, 1.1303122f, 1.0119331f, 0.97557783f, 1.0112488f,
            1.0004334f, -0.016317263f, -0.06595742f, 0.027283654f, -0.00510746f, -0.056057215f, -0.026439143f,
            0.0050763558f };
    private static final float[] LN_INFER = { 0.8777443f, 0.7515294f, 1.5177605f, -1.7520771f, -0.18658507f, -0.74837685f,
            -0.4804688f, 1.403485f, 0.48535234f, -1.3231275f, -0.76324314f, 1.1293601f, -1.0858077f, -0.22025305f,
            0.081564926f, 1.3753421f, -1.6146148f, -0.6455953f, 1.2577467f, -0.16755447f, -0.6879235f, -0.9968435f,
            -0.37706035f, 0.2106614f, -1.477375f, 0.5332682f, 1.8320856f, 0.16712226f, -0.24795188f, 0.95453024f,
            1.7490954f, -0.9680338f, 0.46265042f, -0.4749279f, -1.462945f };

    // one column leaves BatchNorm nothing to normalize, so every output is beta and
    // every input gradient cancels: what this pins is that eps produces no NaN. Two
    // columns make the 2/m in the input gradient exactly 1.0f, which scaleInplace skips
    private static final float[] BN_M1_FORWARD = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    private static final float[] BN_M1_BACKWARD = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    private static final float[] LN_M1_FORWARD = { 0.7734654f, -1.5605973f, -0.19821572f, -0.43355203f, -0.951886f,
            1.2915869f, 1.0791986f };
    private static final float[] LN_M1_BACKWARD = { 0.33654347f, 0.40700057f, -0.2539737f, 0.010755595f, -0.45725504f,
            -0.2804996f, 0.23742872f };
    private static final float[] BN_M2_FORWARD = { -0.99976504f, -0.99999774f, 0.99999297f, -0.9999419f, 0.99991816f,
            0.9999975f, 0.99976504f, 0.99999774f, -0.99999297f, 0.9999419f, -0.99991816f, -0.9999975f };
    private static final float[] BN_M2_BACKWARD = { 4.887581E-5f, -3.2037497E-7f, -9.596348E-6f, 8.440018E-5f,
            -2.580881E-4f, -5.811453E-7f, -4.863739E-5f, 3.2782555E-7f, 9.655952E-6f, -8.440018E-5f, 2.579689E-4f,
            5.811453E-7f };
    private static final float[] LN_M2_FORWARD = { 0.7734654f, -1.5605973f, -0.19821572f, -0.43355203f, -0.951886f,
            1.2915869f, 1.0791986f, 1.8530554f, -1.169496f, 0.67441106f, -0.7415907f, -0.8909677f, -0.28820422f,
            0.5627926f };
    private static final float[] LN_M2_BACKWARD = { 0.33654347f, 0.40700057f, -0.2539737f, 0.010755595f, -0.45725504f,
            -0.2804996f, 0.23742872f, 0.5397484f, 0.56800956f, -0.36343375f, -0.020142935f, 0.12306641f, -0.56490284f,
            -0.2823448f };

    @Test
    void reluIsUnchanged() {
        assertActivation(new Relu(), RELU_FORWARD, RELU_BACKWARD);
    }

    @Test
    void geluIsUnchanged() {
        assertActivation(new Gelu(), GELU_FORWARD, GELU_BACKWARD);
    }

    @Test
    void sigmoidIsUnchanged() {
        assertActivation(new Sigmoid(), SIGMOID_FORWARD, SIGMOID_BACKWARD);
    }

    @Test
    void theDropoutMaskForSeedSevenIsUnchanged() {
        Dropout dropout = new Dropout(0.4f, 7L);
        dropout.setMode(NetworkMode.TRAIN);
        MatrixF ones = Matrices.createF(4, 3);
        Arrays.fill(ones.getArrayUnsafe(), 1.0f);
        assertExactly(MASK_SEED_SEVEN, dropout.forward(ones));
    }

    @Test
    void theGemmFreeChainIsUnchanged() {
        List<Float> losses = new ArrayList<>();
        SigmoidBCELoss loss = new SigmoidBCELoss();
        loss.registerLossCallback(l -> losses.add(Matrices.colsAverage(l).toScalar()));

        Net net = new Net();
        net.add(new Dropout(0.3f, 31L));
        net.add(new Relu());
        net.add(new Dropout(0.25f, 32L));
        net.add(new Gelu());
        net.add(loss);

        MatrixF expected = Matrices.randomUniformF(6, 5, 0.0f, 1.0f, 33L);
        for (int batch = 0; batch < CHAIN_LOSSES.length; ++batch) {
            net.train(Matrices.randomUniformF(6, 5, -2.0f, 2.0f, 40L + batch), expected);
        }

        assertEquals(CHAIN_LOSSES.length, losses.size());
        for (int i = 0; i < CHAIN_LOSSES.length; ++i) {
            assertEquals(CHAIN_LOSSES[i], losses.get(i).floatValue(), "loss of batch " + i);
        }
    }

    @Test
    void batchNormIsUnchanged() throws ReflectiveOperationException {
        BatchNorm bn = new BatchNorm(BN_D);
        bn.setMode(NetworkMode.TRAIN);
        Sgd sgd = GradientCheck.sgdOver(bn, 0.1f);
        // one step first: gamma starts at exactly 1 and beta at 0, so a dropped scale or
        // shift would not show up in the very first output. Each step() has to land where
        // the update used to, at the end of backward: a step after the second forward
        // would let that forward read a gamma of exactly 1 and move four of these arrays.
        bn.forward(normInput(BN_D, BN_M, 31L));
        bn.backward(normGrad(BN_D, BN_M, 32L));
        sgd.step();

        assertExactly(BN_FORWARD, bn.forward(normInput(BN_D, BN_M, 31L)));
        assertExactly(BN_BACKWARD, bn.backward(normGrad(BN_D, BN_M, 32L)));
        sgd.step();

        MatrixF gamma = GradientCheck.value(bn, "gamma");
        MatrixF beta = GradientCheck.value(bn, "beta");
        MatrixF runningMean = GradientCheck.value(bn, "runningMean");
        MatrixF runningVar = GradientCheck.value(bn, "runningVar");
        assertExactly(BN_PARAMS, gamma, beta, runningMean, runningVar);

        bn.setMode(NetworkMode.INFER);
        assertExactly(BN_INFER, bn.forward(normInput(BN_D, BN_M, 31L)));
    }

    @Test
    void layerNormIsUnchanged() throws ReflectiveOperationException {
        LayerNorm ln = new LayerNorm(LN_D);
        ln.setMode(NetworkMode.TRAIN);
        Sgd sgd = GradientCheck.sgdOver(ln, 0.1f);
        ln.forward(normInput(LN_D, LN_M, 33L));
        ln.backward(normGrad(LN_D, LN_M, 34L));
        sgd.step();

        assertExactly(LN_FORWARD, ln.forward(normInput(LN_D, LN_M, 33L)));
        assertExactly(LN_BACKWARD, ln.backward(normGrad(LN_D, LN_M, 34L)));
        sgd.step();

        MatrixF gamma = GradientCheck.value(ln, "gamma");
        MatrixF beta = GradientCheck.value(ln, "beta");
        assertExactly(LN_PARAMS, gamma, beta);

        ln.setMode(NetworkMode.INFER);
        assertExactly(LN_INFER, ln.forward(normInput(LN_D, LN_M, 33L)));
    }

    @Test
    void theNormLayersAtOneAndTwoColumnsAreUnchanged() {
        assertNormEdge(new BatchNorm(BN_D), BN_D, 1, BN_M1_FORWARD, BN_M1_BACKWARD);
        assertNormEdge(new LayerNorm(LN_D), LN_D, 1, LN_M1_FORWARD, LN_M1_BACKWARD);
        assertNormEdge(new BatchNorm(BN_D), BN_D, 2, BN_M2_FORWARD, BN_M2_BACKWARD);
        assertNormEdge(new LayerNorm(LN_D), LN_D, 2, LN_M2_FORWARD, LN_M2_BACKWARD);
    }

    private static void assertNormEdge(Layer layer, int rows, int cols, float[] forward, float[] backward) {
        layer.setMode(NetworkMode.TRAIN);
        assertExactly(forward, layer.forward(normInput(rows, cols, 35L)));
        assertExactly(backward, layer.backward(normGrad(rows, cols, 36L)));
    }

    private static MatrixF normInput(int rows, int cols, long seed) {
        return Matrices.randomUniformF(rows, cols, -2.0f, 2.0f, seed);
    }

    private static MatrixF normGrad(int rows, int cols, long seed) {
        return Matrices.randomUniformF(rows, cols, -1.0f, 1.0f, seed);
    }

    private static void assertActivation(Activation activation, float[] forward, float[] backward) {
        MatrixF x = Matrices.randomUniformF(4, 3, -2.0f, 2.0f, 21L);
        MatrixF g = Matrices.randomUniformF(4, 3, -1.0f, 1.0f, 22L);
        activation.setMode(NetworkMode.TRAIN);
        assertExactly(forward, activation.forward(x.copy()));
        assertExactly(backward, activation.backward(g.copy()));
    }

    /** Several matrices are matched against one array, laid out end to end. */
    private static void assertExactly(float[] expected, MatrixF... actual) {
        int i = 0;
        for (MatrixF m : actual) {
            for (float value : m.getArrayUnsafe()) {
                assertEquals(expected[i], value, "element " + i);
                ++i;
            }
        }
        assertEquals(expected.length, i);
    }

    private static final class Net extends AbstractNetwork {
    }
}
