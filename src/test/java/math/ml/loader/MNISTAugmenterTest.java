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
package math.ml.loader;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

public class MNISTAugmenterTest {

    private static final int W = 28;
    private static final int H = 28;

    private static byte[] ramp() {
        byte[] img = new byte[W * H];
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                // asymmetric in both axes, so a swapped or mirrored axis cannot pass
                img[y * W + x] = (byte) ((3 * x + 7 * y) % 256);
            }
        }
        return img;
    }

    private static byte[] centeredBlock(int size, int value) {
        byte[] img = new byte[W * H];
        int from = (W - size) / 2;
        for (int y = from; y < from + size; ++y) {
            for (int x = from; x < from + size; ++x) {
                img[y * W + x] = (byte) value;
            }
        }
        return img;
    }

    private static long sum(byte[] img) {
        long s = 0L;
        for (byte b : img) {
            s += b & 0xFF;
        }
        return s;
    }

    @Test
    public void identityIsExact() {
        byte[] src = ramp();
        assertArrayEquals(src, MNISTAugmenter.warp(src, W, H, 0.0, 0.0, 0.0, 1.0));
    }

    @Test
    public void wholePixelTranslationShiftsIndices() {
        byte[] src = ramp();
        byte[] dst = MNISTAugmenter.warp(src, W, H, 1.0, 0.0, 0.0, 1.0);
        for (int y = 0; y < H; ++y) {
            assertEquals(0, dst[y * W] & 0xFF, "column 0 must be carried-in background");
            for (int x = 1; x < W; ++x) {
                assertEquals(src[y * W + x - 1] & 0xFF, dst[y * W + x] & 0xFF);
            }
        }
    }

    @Test
    public void wholePixelTranslationMovesDownwards() {
        byte[] src = ramp();
        byte[] dst = MNISTAugmenter.warp(src, W, H, 0.0, 1.0, 0.0, 1.0);
        for (int x = 0; x < W; ++x) {
            assertEquals(0, dst[x] & 0xFF);
        }
        for (int y = 1; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                assertEquals(src[(y - 1) * W + x] & 0xFF, dst[y * W + x] & 0xFF);
            }
        }
    }

    @Test
    public void halfPixelTranslationAveragesNeighbours() {
        byte[] src = ramp();
        byte[] dst = MNISTAugmenter.warp(src, W, H, 0.5, 0.0, 0.0, 1.0);
        for (int y = 0; y < H; ++y) {
            for (int x = 1; x < W; ++x) {
                int expected = Math.round(0.5f * (src[y * W + x - 1] & 0xFF) + 0.5f * (src[y * W + x] & 0xFF));
                assertEquals(expected, dst[y * W + x] & 0xFF, 1);
            }
        }
    }

    @Test
    public void quarterTurnRotatesTheGrid() {
        byte[] src = ramp();
        byte[] dst = MNISTAugmenter.warp(src, W, H, 0.0, 0.0, Math.PI / 2.0, 1.0);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                assertEquals(src[(W - 1 - x) * W + y] & 0xFF, dst[y * W + x] & 0xFF, 1);
            }
        }
    }

    @Test
    public void fullTurnIsIdentity() {
        byte[] src = ramp();
        byte[] dst = MNISTAugmenter.warp(src, W, H, 0.0, 0.0, 2.0 * Math.PI, 1.0);
        for (int i = 0; i < src.length; ++i) {
            assertEquals(src[i] & 0xFF, dst[i] & 0xFF, 1);
        }
    }

    @Test
    public void translationOffTheGridYieldsZero() {
        byte[] dst = MNISTAugmenter.warp(ramp(), W, H, 100.0, 0.0, 0.0, 1.0);
        assertEquals(0L, sum(dst));
    }

    @Test
    public void enlargingScalesTheArea() {
        byte[] src = centeredBlock(8, 255);
        double scale = 1.25;
        byte[] dst = MNISTAugmenter.warp(src, W, H, 0.0, 0.0, 0.0, scale);
        double ratio = sum(dst) / (double) sum(src);
        assertEquals(scale * scale, ratio, 0.1, "mass should grow with the square of the scale");
    }

    @Test
    public void shrinkingScalesTheArea() {
        byte[] src = centeredBlock(12, 255);
        double scale = 0.8;
        byte[] dst = MNISTAugmenter.warp(src, W, H, 0.0, 0.0, 0.0, scale);
        double ratio = sum(dst) / (double) sum(src);
        assertEquals(scale * scale, ratio, 0.1);
    }

    @Test
    public void extremeParametersStayInByteRange() {
        byte[] src = new byte[W * H];
        Arrays.fill(src, (byte) 255);
        byte[] dst = MNISTAugmenter.warp(src, W, H, -1.7, 1.9, Math.toRadians(-12.0), 1.1);
        for (byte b : dst) {
            int v = b & 0xFF;
            assertTrue(v >= 0 && v <= 255);
        }
    }

    @Test
    public void isDeterministic() {
        byte[] src = ramp();
        byte[] a = MNISTAugmenter.warp(src, W, H, -0.3, 1.4, 0.17, 0.94);
        byte[] b = MNISTAugmenter.warp(src, W, H, -0.3, 1.4, 0.17, 0.94);
        assertArrayEquals(a, b);
    }

    @Test
    public void rejectsAMismatchedLength() {
        assertThrows(IllegalArgumentException.class, () -> MNISTAugmenter.warp(new byte[10], W, H, 0.0, 0.0, 0.0, 1.0));
    }

    @Test
    public void rejectsANonPositiveScale() {
        byte[] src = ramp();
        assertThrows(IllegalArgumentException.class, () -> MNISTAugmenter.warp(src, W, H, 0.0, 0.0, 0.0, 0.0));
    }

    private static double[] constantField(double value) {
        double[] f = new double[W * H];
        Arrays.fill(f, value);
        return f;
    }

    private static double variance(double[] f) {
        double mean = 0.0;
        for (double v : f) {
            mean += v;
        }
        mean /= f.length;
        double s = 0.0;
        for (double v : f) {
            s += (v - mean) * (v - mean);
        }
        return s / f.length;
    }

    @Test
    public void zeroFieldIsIdentity() {
        byte[] src = ramp();
        assertArrayEquals(src, MNISTAugmenter.warpField(src, W, H, constantField(0.0), constantField(0.0)));
    }

    @Test
    public void constantFieldEqualsTheOppositeTranslation() {
        // warp moves the image by dx, warpField says where each pixel reads from, so the
        // two agree only with the sign flipped -- which is the point of the check
        byte[] src = ramp();
        byte[] viaField = MNISTAugmenter.warpField(src, W, H, constantField(0.75), constantField(-1.5));
        byte[] viaWarp = MNISTAugmenter.warp(src, W, H, -0.75, 1.5, 0.0, 1.0);
        assertArrayEquals(viaWarp, viaField);
    }

    @Test
    public void fieldOffTheGridYieldsZero() {
        byte[] dst = MNISTAugmenter.warpField(ramp(), W, H, constantField(100.0), constantField(0.0));
        assertEquals(0L, sum(dst));
    }

    @Test
    public void rejectsAMismatchedField() {
        byte[] src = ramp();
        assertThrows(IllegalArgumentException.class,
                () -> MNISTAugmenter.warpField(src, W, H, new double[10], constantField(0.0)));
    }

    @Test
    public void smoothingKeepsTheCentreAndDampensTheBorder() {
        double[] f = constantField(1.0);
        MNISTAugmenter.smooth(f, W, H, MNISTAugmenter.gaussianKernel(4.0));
        // the kernel is normalized, so a constant field survives where it fits entirely
        assertEquals(1.0, f[(H / 2) * W + W / 2], 1.0e-9);
        // and is damped where it does not, because outside the grid counts as zero
        assertTrue(f[0] < 0.5, "corner was " + f[0]);
    }

    @Test
    public void aLargerSigmaSmoothsMore() {
        double[] narrow = MNISTAugmenter.displacementField(W, H, MNISTAugmenter.gaussianKernel(3.0), 1.0,
                new SplittableRandom(7L));
        double[] wide = MNISTAugmenter.displacementField(W, H, MNISTAugmenter.gaussianKernel(8.0), 1.0,
                new SplittableRandom(7L));
        assertTrue(variance(wide) < variance(narrow),
                "wide " + variance(wide) + " should be below narrow " + variance(narrow));
    }

    @Test
    public void displacementScalesWithAlpha() {
        double[] one = MNISTAugmenter.displacementField(W, H, MNISTAugmenter.gaussianKernel(4.0), 1.0,
                new SplittableRandom(7L));
        double[] ten = MNISTAugmenter.displacementField(W, H, MNISTAugmenter.gaussianKernel(4.0), 10.0,
                new SplittableRandom(7L));
        for (int i = 0; i < one.length; ++i) {
            assertEquals(10.0 * one[i], ten[i], 1.0e-9);
        }
    }

    @Test
    public void elasticWithoutAmplitudeIsIdentity() {
        byte[] src = ramp();
        assertArrayEquals(src, MNISTAugmenter.elastic(src, W, H, 4.0, 0.0, new SplittableRandom(7L)));
    }

    @Test
    public void elasticIsReproducible() {
        byte[] src = ramp();
        byte[] a = MNISTAugmenter.elastic(src, W, H, 4.0, 34.0, new SplittableRandom(7L));
        byte[] b = MNISTAugmenter.elastic(src, W, H, 4.0, 34.0, new SplittableRandom(7L));
        assertArrayEquals(a, b);
    }

    @Test
    public void elasticRejectsANonPositiveSigma() {
        byte[] src = ramp();
        assertThrows(IllegalArgumentException.class,
                () -> MNISTAugmenter.elastic(src, W, H, 0.0, 34.0, new SplittableRandom(7L)));
    }

    private static byte[] elasticWith(byte[] src, long seed, double sigma, double alpha) {
        SplittableRandom rnd = new SplittableRandom(seed);
        // randomElastic spends one boolean on the choice before it distorts anything
        rnd.nextBoolean();
        return MNISTAugmenter.elastic(src, W, H, sigma, alpha, rnd);
    }

    @Test
    public void randomAffineKeepsTheDigitOnTheGrid() {
        // 8 pixels wide and centered: at scale 1.1, 12 degrees and 2 pixels of shift its
        // farthest corner still lands well inside the frame, so nothing may be clipped away
        byte[] src = centeredBlock(8, 200);
        long expected = sum(src);
        int moved = 0;
        for (long seed = 0L; seed < 200L; ++seed) {
            byte[] got = MNISTAugmenter.randomAffine(src, W, H, new SplittableRandom(seed));
            long ink = sum(got);
            assertTrue(ink > expected / 2L && ink < 2L * expected, "seed " + seed + " summed " + ink);
            if (!Arrays.equals(src, got)) {
                ++moved;
            }
        }
        assertEquals(200, moved);
    }

    @Test
    public void randomAffineIsReproducible() {
        byte[] src = ramp();
        assertArrayEquals(MNISTAugmenter.randomAffine(src, W, H, new SplittableRandom(7L)),
                MNISTAugmenter.randomAffine(src, W, H, new SplittableRandom(7L)));
    }

    @Test
    public void randomElasticIsReproducible() {
        byte[] src = ramp();
        assertArrayEquals(MNISTAugmenter.randomElastic(src, W, H, new SplittableRandom(7L)),
                MNISTAugmenter.randomElastic(src, W, H, new SplittableRandom(7L)));
    }

    @Test
    public void randomElasticUsesBothCalibrations() {
        byte[] src = ramp();
        int first = 0;
        int second = 0;
        for (long seed = 0L; seed < 40L; ++seed) {
            byte[] got = MNISTAugmenter.randomElastic(src, W, H, new SplittableRandom(seed));
            if (Arrays.equals(got,
                    elasticWith(src, seed, MNISTAugmenter.ELASTIC1_SIGMA, MNISTAugmenter.ELASTIC1_ALPHA))) {
                ++first;
            } else if (Arrays.equals(got,
                    elasticWith(src, seed, MNISTAugmenter.ELASTIC2_SIGMA, MNISTAugmenter.ELASTIC2_ALPHA))) {
                ++second;
            }
        }
        assertEquals(40, first + second, "every draw must be one of the two calibrations");
        assertTrue(first > 0 && second > 0, "first " + first + ", second " + second);
    }

}
