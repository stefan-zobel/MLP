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
 * Writes affinely distorted copies of the MNIST training images, one independent draw
 * per image.
 */
public final class MNISTAugmenter {

    private static final String TRAIN_IMAGES = "./data/mnist/train-images.idx3-ubyte";

    /** The sets this generator writes, in the order the seeds are drawn for them. */
    private static final String[] TARGETS = { "./data/mnist/train-images-affine1.idx3-ubyte",
            "./data/mnist/train-images-affine2.idx3-ubyte" };

    private static final int IMAGE_MAGIC = 0x00000803;

    private static final double MAX_SHIFT = 2.0;
    private static final double MAX_ANGLE = Math.toRadians(12.0);
    private static final double MIN_SCALE = 0.9;
    private static final double MAX_SCALE = 1.1;

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
     * Writes the augmented sets named in {@code TARGETS}.
     *
     * @param args optionally the base seed to reproduce an earlier run
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
            for (String target : TARGETS) {
                // one generator per file, so a file may be regenerated on its own and the
                // draws for the other one are unaffected
                write(target, source, new SplittableRandom(seeds.nextLong()));
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static void write(String target, Images source, SplittableRandom rnd) throws IOException {
        final int pixels = source.w * source.h;
        // a half-written file still has a valid-looking header, so the name only appears
        // once the whole set is on disk
        Path tmp = Path.of(target + ".tmp");
        byte[] image = new byte[pixels];
        try (DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(new FileOutputStream(tmp.toFile())))) {
            out.writeInt(IMAGE_MAGIC);
            out.writeInt(source.count);
            out.writeInt(source.h);
            out.writeInt(source.w);
            for (int i = 0; i < source.count; ++i) {
                System.arraycopy(source.data, i * pixels, image, 0, pixels);
                double dx = rnd.nextDouble(-MAX_SHIFT, MAX_SHIFT);
                double dy = rnd.nextDouble(-MAX_SHIFT, MAX_SHIFT);
                double theta = rnd.nextDouble(-MAX_ANGLE, MAX_ANGLE);
                double scale = rnd.nextDouble(MIN_SCALE, MAX_SCALE);
                out.write(warp(image, source.w, source.h, dx, dy, theta, scale));
            }
        }
        Files.move(tmp, Path.of(target), StandardCopyOption.REPLACE_EXISTING);
        System.out.println("wrote " + source.count + " images to " + target);
    }

    private static Images read(String path) throws IOException {
        try (DataInputStream in = MNIST.getDataInputStream(path)) {
            int count = in.readInt();
            int rows = in.readInt();
            int cols = in.readInt();
            byte[] data = new byte[count * rows * cols];
            in.readFully(data);
            return new Images(count, cols, rows, data);
        }
    }

    // the whole source set, row-major per image, unsigned bytes
    private static final class Images {
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
