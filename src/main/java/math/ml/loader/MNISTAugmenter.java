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

import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.SplittableRandom;

/**
 * Writes distorted copies of the MNIST training images, one independent draw per image.
 */
public final class MNISTAugmenter {

    static final String TRAIN_IMAGES = "./data/mnist/train-images.idx3-ubyte";

    private static final int IMAGE_MAGIC = 0x00000803;

    private static final double MAX_SHIFT = 2.0;
    private static final double MAX_ANGLE = Math.toRadians(12.0);
    private static final double MIN_SCALE = 0.9;
    private static final double MAX_SCALE = 1.1;

    // Calibrated against a net that scores 1.0 on the undistorted training images and
    // 0.910 on the two affine sets: sigma 4 / alpha 34 leaves it at 0.842, sigma 6 /
    // alpha 44 at about 0.91. Difficulty follows the displacement almost alone -- 1.3 px
    // RMS for the first pair, 1.1 px for the second -- while sigma decides whether that
    // displacement wrinkles locally or moves whole regions, so the two sets cover both.
    static final double ELASTIC1_SIGMA = 4.0;
    static final double ELASTIC1_ALPHA = 34.0;
    static final double ELASTIC2_SIGMA = 6.0;
    static final double ELASTIC2_ALPHA = 44.0;

    /**
     * The sets this generator writes, in the order the seeds are drawn for them. New
     * entries belong at the end: the order is what fixes which draws a file gets, so
     * appending leaves the earlier files reproducible from the same base seed.
     */
    private static final Target[] TARGETS = { new Target("affine1", MNISTAugmenter::randomAffine),
            new Target("affine2", MNISTAugmenter::randomAffine),
            new Target("elastic1", (image, w, h, rnd) -> elastic(image, w, h, ELASTIC1_SIGMA, ELASTIC1_ALPHA, rnd)),
            new Target("elastic2", (image, w, h, rnd) -> elastic(image, w, h, ELASTIC2_SIGMA, ELASTIC2_ALPHA, rnd)) };


    /**
     * Resamples an image under a rotation, an isotropic scaling and a translation, all
     * about the image center.
     * <p>
     * The map is evaluated backwards - for every destination pixel the source coordinate
     * is computed and sampled bilinearly - because a forward map leaves holes wherever
     * the transformation spreads neighboring pixels apart. Samples outside the source
     * grid count as zero, so a translation carries in background rather than an edge.
     *
     * @param src   source image, row-major, {@code w * h} unsigned bytes
     * @param w     image width in pixels
     * @param h     image height in pixels
     * @param dx    horizontal translation in pixels, positive to the right, need not be
     *              a whole number
     * @param dy    vertical translation in pixels, positive downwards
     * @param theta rotation angle in radians
     * @param scale isotropic scale factor, {@code > 1} enlarges the image
     * @return the resampled image, row-major, {@code w * h} unsigned bytes
     */
    public static byte[] warp(byte[] src, int w, int h, double dx, double dy, double theta, double scale) {
        if (src.length != w * h) {
            throw new IllegalArgumentException("src.length " + src.length + " != " + w + " * " + h);
        }
        if (scale <= 0.0) {
            throw new IllegalArgumentException("scale must be positive: " + scale);
        }
        final double cx = (w - 1) / 2.0;
        final double cy = (h - 1) / 2.0;
        final double cos = Math.cos(theta);
        final double sin = Math.sin(theta);
        byte[] dst = new byte[src.length];
        for (int yd = 0; yd < h; ++yd) {
            final double v = yd - cy - dy;
            for (int xd = 0; xd < w; ++xd) {
                double u = xd - cx - dx;
                double xs = cx + (cos * u + sin * v) / scale;
                double ys = cy + (-sin * u + cos * v) / scale;
                dst[yd * w + xd] = (byte) sample(src, w, h, xs, ys);
            }
        }
        return dst;
    }

    // bilinear, zero outside the grid, rounded into [0, 255]
    private static int sample(byte[] src, int w, int h, double x, double y) {
        int x0 = (int) Math.floor(x);
        int y0 = (int) Math.floor(y);
        double fx = x - x0;
        double fy = y - y0;
        double top = (1.0 - fx) * at(src, w, h, x0, y0) + fx * at(src, w, h, x0 + 1, y0);
        double bottom = (1.0 - fx) * at(src, w, h, x0, y0 + 1) + fx * at(src, w, h, x0 + 1, y0 + 1);
        long value = Math.round((1.0 - fy) * top + fy * bottom);
        return (int) Math.min(255L, Math.max(0L, value));
    }

    private static int at(byte[] src, int w, int h, int x, int y) {
        if (x < 0 || x >= w || y < 0 || y >= h) {
            return 0;
        }
        return src[y * w + x] & 0xFF;
    }

    /**
     * Resamples an image under an explicit per-pixel displacement,
     * {@code dst(x, y) = src(x + dispX, y + dispY)}, sampled the way {@link #warp} samples.
     *
     * @param src   source image, row-major, {@code w * h} unsigned bytes
     * @param w     image width in pixels
     * @param h     image height in pixels
     * @param dispX horizontal displacement per destination pixel, row-major
     * @param dispY vertical displacement per destination pixel, row-major
     * @return the resampled image, row-major, {@code w * h} unsigned bytes
     */
    public static byte[] warpField(byte[] src, int w, int h, double[] dispX, double[] dispY) {
        if (src.length != w * h) {
            throw new IllegalArgumentException("src.length " + src.length + " != " + w + " * " + h);
        }
        if (dispX.length != src.length || dispY.length != src.length) {
            throw new IllegalArgumentException("the displacement fields must match the image");
        }
        byte[] dst = new byte[src.length];
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int i = y * w + x;
                dst[i] = (byte) sample(src, w, h, x + dispX[i], y + dispY[i]);
            }
        }
        return dst;
    }

    /**
     * Distorts an image with a smoothed random displacement field - the local deformation
     * that no affine map can produce.
     *
     * @param src   source image, row-major, {@code w * h} unsigned bytes
     * @param w     image width in pixels
     * @param h     image height in pixels
     * @param sigma width of the Gaussian smoothing the field; small values wrinkle
     *              locally, large ones move whole regions together
     * @param alpha scale applied after the smoothing, in pixels
     * @param rnd   source of the two displacement fields
     * @return the distorted image, row-major, {@code w * h} unsigned bytes
     */
    public static byte[] elastic(byte[] src, int w, int h, double sigma, double alpha, SplittableRandom rnd) {
        double[] kernel = gaussianKernel(sigma);
        return warpField(src, w, h, displacementField(w, h, kernel, alpha, rnd),
                displacementField(w, h, kernel, alpha, rnd));
    }

    static double[] displacementField(int w, int h, double[] kernel, double alpha, SplittableRandom rnd) {
        double[] field = new double[w * h];
        for (int i = 0; i < field.length; ++i) {
            field[i] = rnd.nextDouble(-1.0, 1.0);
        }
        smooth(field, w, h, kernel);
        for (int i = 0; i < field.length; ++i) {
            field[i] *= alpha;
        }
        return field;
    }

    // Separable and in place. The kernel sums to one but is not renormalized at the
    // border, where the field counts as zero -- that is what the reference
    // implementations do, and together with the normalization it is the reason alpha has
    // to be as large as it is to move a pixel at all.
    static void smooth(double[] field, int w, int h, double[] kernel) {
        final int r = (kernel.length - 1) / 2;
        double[] tmp = new double[field.length];
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                double s = 0.0;
                for (int i = -r; i <= r; ++i) {
                    int xx = x + i;
                    if (xx >= 0 && xx < w) {
                        s += kernel[i + r] * field[y * w + xx];
                    }
                }
                tmp[y * w + x] = s;
            }
        }
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                double s = 0.0;
                for (int i = -r; i <= r; ++i) {
                    int yy = y + i;
                    if (yy >= 0 && yy < h) {
                        s += kernel[i + r] * tmp[yy * w + x];
                    }
                }
                field[y * w + x] = s;
            }
        }
    }

    static double[] gaussianKernel(double sigma) {
        if (sigma <= 0.0) {
            throw new IllegalArgumentException("sigma must be positive: " + sigma);
        }
        final int r = (int) Math.ceil(3.0 * sigma);
        double[] k = new double[2 * r + 1];
        double sum = 0.0;
        for (int i = -r; i <= r; ++i) {
            k[i + r] = Math.exp(-(i * i) / (2.0 * sigma * sigma));
            sum += k[i + r];
        }
        for (int i = 0; i < k.length; ++i) {
            k[i] /= sum;
        }
        return k;
    }

    /**
     * Draws one affine distortion - a translation, a rotation and an isotropic scaling -
     * and applies it. These are the parameters the stored affine sets were written with.
     *
     * @param image source image, row-major, {@code w * h} unsigned bytes
     * @param w     image width in pixels
     * @param h     image height in pixels
     * @param rnd   source of the four parameters
     * @return the distorted image, row-major, {@code w * h} unsigned bytes
     */
    // the draw order fixes the contents of the affine sets and must not be rearranged
    public static byte[] randomAffine(byte[] image, int w, int h, SplittableRandom rnd) {
        double dx = rnd.nextDouble(-MAX_SHIFT, MAX_SHIFT);
        double dy = rnd.nextDouble(-MAX_SHIFT, MAX_SHIFT);
        double theta = rnd.nextDouble(-MAX_ANGLE, MAX_ANGLE);
        double scale = rnd.nextDouble(MIN_SCALE, MAX_SCALE);
        return warp(image, w, h, dx, dy, theta, scale);
    }

    /**
     * Draws one of the two calibrated elastic distortions with equal probability and
     * applies it.
     *
     * @param image source image, row-major, {@code w * h} unsigned bytes
     * @param w     image width in pixels
     * @param h     image height in pixels
     * @param rnd   source of the choice and of the two displacement fields
     * @return the distorted image, row-major, {@code w * h} unsigned bytes
     */
    public static byte[] randomElastic(byte[] image, int w, int h, SplittableRandom rnd) {
        if (rnd.nextBoolean()) {
            return elastic(image, w, h, ELASTIC1_SIGMA, ELASTIC1_ALPHA, rnd);
        }
        return elastic(image, w, h, ELASTIC2_SIGMA, ELASTIC2_ALPHA, rnd);
    }

    /**
     * Writes the augmented sets.
     *
     * @param args optionally the base seed to reproduce an earlier run, then the names of
     *             the sets to write; no names means all of them
     */
    public static void main(String[] args) {
        // pass this seed back as the first argument to repeat a run exactly
        long baseSeed = args.length > 0 ? Long.parseLong(args[0]) : new SplittableRandom().nextLong();
        System.out.println("seed: " + baseSeed);
        SplittableRandom seeds = new SplittableRandom(baseSeed);
        try {
            Images source = read(TRAIN_IMAGES);
            System.out.println("read " + source.count + " images of " + source.w + " x " + source.h + " from "
                    + TRAIN_IMAGES);
            for (Target target : TARGETS) {
                // one generator per file, and drawn for every target whether or not it is
                // selected: skipping a draw here would silently change what a later file
                // contains, which is the one thing the base seed has to protect
                SplittableRandom rnd = new SplittableRandom(seeds.nextLong());
                if (selected(args, target.name)) {
                    write(target, source, rnd);
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static boolean selected(String[] args, String name) {
        if (args.length <= 1) {
            return true;
        }
        for (int i = 1; i < args.length; ++i) {
            if (name.equals(args[i])) {
                return true;
            }
        }
        return false;
    }

    private static void write(Target target, Images source, SplittableRandom rnd) throws IOException {
        final int pixels = source.w * source.h;
        // a half-written file still has a valid-looking header, so the name only appears
        // once the whole set is on disk
        Path tmp = Path.of(target.path + ".tmp");
        byte[] image = new byte[pixels];
        try (DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(new FileOutputStream(tmp.toFile())))) {
            out.writeInt(IMAGE_MAGIC);
            out.writeInt(source.count);
            out.writeInt(source.h);
            out.writeInt(source.w);
            for (int i = 0; i < source.count; ++i) {
                System.arraycopy(source.data, i * pixels, image, 0, pixels);
                out.write(target.distortion.apply(image, source.w, source.h, rnd));
            }
        }
        Files.move(tmp, Path.of(target.path), StandardCopyOption.REPLACE_EXISTING);
        System.out.println("wrote " + source.count + " images to " + target.path);
    }

    // package-private because MNISTAugmentedSet needs the raw bytes
    static Images read(String path) throws IOException {
        try (DataInputStream in = MNIST.getDataInputStream(path)) {
            int count = in.readInt();
            int rows = in.readInt();
            int cols = in.readInt();
            byte[] data = new byte[count * rows * cols];
            in.readFully(data);
            return new Images(count, cols, rows, data);
        }
    }

    // what one output set does to a single image
    private interface Distortion {
        byte[] apply(byte[] image, int w, int h, SplittableRandom rnd);
    }

    private static final class Target {
        final String name;
        final String path;
        final Distortion distortion;

        Target(String name, Distortion distortion) {
            this.name = name;
            this.path = "./data/mnist/train-images-" + name + ".idx3-ubyte";
            this.distortion = distortion;
        }
    }

    // the whole source set, row-major per image, unsigned bytes
    static final class Images {
        final int count;
        final int w;
        final int h;
        final byte[] data;

        Images(int count, int w, int h, byte[] data) {
            this.count = count;
            this.w = w;
            this.h = h;
            this.data = data;
        }
    }

    private MNISTAugmenter() {
        throw new AssertionError();
    }
}
