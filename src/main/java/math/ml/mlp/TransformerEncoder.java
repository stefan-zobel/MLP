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

import java.io.IOException;

import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;
import java.util.SplittableRandom;
import java.util.function.Supplier;

import net.jamu.matrix.MatrixF;

/**
 * A vision transformer encoder: flat images in, one pooled vector per image out.
 *
 * <p>The patch projection is a {@link Conv2D} whose stride equals its kernel, which is the same
 * operation as a linear projection of each tile and already emits its columns sample-major,
 * position-minor -- the layout {@link Attention} and {@link MeanPool} need. The sequence length
 * follows from the geometry and is read back rather than passed in.
 *
 * <p>Both ends speak the ordinary {@code features x batch} convention, so the folded token layout
 * never leaves this layer. What goes on top of the pooled vector -- a classification head, a
 * regression head, nothing at all -- is the caller's choice.
 *
 * <p>Built through {@link #builder()}, because the geometry and the model dimensions are a dozen
 * interchangeable integers and a positional list of those is hard to read and easy to transpose.
 */
public final class TransformerEncoder extends AbstractLayer {

    private final List<Layer> layers;
    private final int seqLen;
    private final int dModel;

    private TransformerEncoder(Builder b) {
        Conv2D patch = new Conv2D(b.channels, b.dModel, b.imageHeight, b.imageWidth, b.tile, b.tile, 0,
                b.names + "patch", b.init, b.seeds.nextLong());
        seqLen = patch.outputHeight() * patch.outputWidth();
        dModel = b.dModel;
        int mlpWidth = b.mlpWidth > 0 ? b.mlpWidth : 4 * b.dModel;

        List<Layer> all = new ArrayList<>();
        all.add(new Unflatten(b.channels, b.imageHeight, b.imageWidth));
        all.add(patch);
        all.add(new PositionalEncoding(b.dModel, seqLen, b.names + "pos", b.seeds.nextLong()));
        for (int i = 0; i < b.blocks; ++i) {
            // pre-norm: the normalization sits inside the branch, so the identity path runs
            // unnormalized from the patch projection all the way to the pooling
            all.add(new ResidualBranch(new LayerNorm(b.dModel, b.names + "b" + i + "_ln1"),
                    new Attention(b.dModel, seqLen, b.heads, b.names + "b" + i + "_attn", b.init,
                            b.seeds.nextLong())));
            all.add(new ResidualBranch(new LayerNorm(b.dModel, b.names + "b" + i + "_ln2"),
                    new Hidden(b.dModel, mlpWidth, b.names + "b" + i + "_fc1", Init.HE,
                            b.seeds.nextLong()),
                    b.activation.get(),
                    new Hidden(mlpWidth, b.dModel, b.names + "b" + i + "_fc2", b.init,
                            b.seeds.nextLong())));
        }
        all.add(new LayerNorm(b.dModel, b.names + "lnf"));
        all.add(new MeanPool(seqLen));
        layers = all;
    }

    /**
     * Starts building an encoder. Image, tile, model width, heads, blocks, names and seed have to
     * be given; everything else has a default.
     *
     * @return a fresh builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** Tokens per image, a consequence of the image and tile sizes rather than a setting. */
    public int sequenceLength() {
        return seqLen;
    }

    /** Rows of the pooled output, which is what a head on top has to be sized for. */
    public int features() {
        return dModel;
    }

    @Override
    public MatrixF forward(MatrixF input) {
        // no super.forward: this layer has no parameters of its own and every inner layer
        // caches whatever it needs
        MatrixF x = input;
        for (Layer layer : layers) {
            x = layer.forward(x);
        }
        return x;
    }

    @Override
    public MatrixF backward(MatrixF grads) {
        if (mode == NetworkMode.INFER) {
            return null;
        }
        MatrixF g = grads;
        for (ListIterator<Layer> it = layers.listIterator(layers.size()); it.hasPrevious();) {
            g = it.previous().backward(g);
        }
        return g;
    }

    @Override
    public void setMode(NetworkMode mode) {
        super.setMode(mode);
        for (Layer layer : layers) {
            layer.setMode(mode);
        }
    }

    /** Only the first layer ever sees the caller's matrix; the rest see intermediate buffers. */
    @Override
    public boolean mutatesInput() {
        return layers.get(0).mutatesInput();
    }

    /** Likewise backward, where the last layer is the one handed the caller's gradients. */
    @Override
    public boolean mutatesGradients() {
        return layers.get(layers.size() - 1).mutatesGradients();
    }

    @Override
    public List<Parameter> parameters() {
        List<Parameter> all = new ArrayList<>();
        for (Layer layer : layers) {
            all.addAll(layer.parameters());
        }
        return all;
    }

    @Override
    public void writeParameters(ParameterSink sink) throws IOException {
        for (Layer layer : layers) {
            layer.writeParameters(sink);
        }
    }

    @Override
    public void readParameters(ParameterSource source) throws IOException {
        for (Layer layer : layers) {
            layer.readParameters(source);
        }
    }

    /** Collects the settings of a {@link TransformerEncoder} and checks them in {@link #build()}. */
    public static final class Builder {

        private int imageHeight = -1;
        private int imageWidth = -1;
        private int channels = 1;
        private int tile = -1;
        private int dModel = -1;
        private int heads = -1;
        private int blocks = -1;
        private int mlpWidth = -1;
        private Supplier<Layer> activation = Relu::new;
        private Init init = Init.GLOROT;
        private String names;
        private SplittableRandom seeds;

        private Builder() {
        }

        /**
         * @param height rows of one image
         * @param width  columns of one image
         * @return this builder
         */
        public Builder image(int height, int width) {
            imageHeight = height;
            imageWidth = width;
            return this;
        }

        /**
         * @param count channels per image, one by default
         * @return this builder
         */
        public Builder channels(int count) {
            channels = count;
            return this;
        }

        /**
         * @param edge edge of a square patch, used as both the kernel and the stride
         * @return this builder
         */
        public Builder tile(int edge) {
            tile = edge;
            return this;
        }

        /**
         * @param features features per token
         * @return this builder
         */
        public Builder dModel(int features) {
            dModel = features;
            return this;
        }

        /**
         * @param count attention heads per block; must divide the model width
         * @return this builder
         */
        public Builder heads(int count) {
            heads = count;
            return this;
        }

        /**
         * @param count attention and feed-forward pairs
         * @return this builder
         */
        public Builder blocks(int count) {
            blocks = count;
            return this;
        }

        /**
         * @param width hidden width of the feed-forward part, four times the model width by default
         * @return this builder
         */
        public Builder mlpWidth(int width) {
            mlpWidth = width;
            return this;
        }

        /**
         * @param supplier supplies the feed-forward activation, one instance per block; ReLU by default
         * @return this builder
         */
        public Builder activation(Supplier<Layer> supplier) {
            activation = supplier;
            return this;
        }

        /**
         * @param scheme weight initialization, Glorot by default
         * @return this builder
         */
        public Builder init(Init scheme) {
            init = scheme;
            return this;
        }

        /**
         * @param prefix prepended to every layer name, which is what namespaces the bundle entries
         * @return this builder
         */
        public Builder names(String prefix) {
            names = prefix;
            return this;
        }

        /**
         * @param seed one seed for the whole encoder, fanned out in layer order
         * @return this builder
         */
        public Builder seed(long seed) {
            seeds = new SplittableRandom(seed);
            return this;
        }

        /**
         * @return the encoder
         * @throws IllegalStateException    if a setting without a default was not given
         * @throws IllegalArgumentException if the settings do not fit together
         */
        public TransformerEncoder build() {
            require(imageHeight > 0 && imageWidth > 0, "image");
            require(tile > 0, "tile");
            require(dModel > 0, "dModel");
            require(heads > 0, "heads");
            require(blocks > 0, "blocks");
            require(names != null, "names");
            require(seeds != null, "seed");
            if (channels <= 0) {
                throw new IllegalArgumentException("channels " + channels + " must be positive");
            }
            if (mlpWidth != -1 && mlpWidth <= 0) {
                throw new IllegalArgumentException("mlpWidth " + mlpWidth + " must be positive");
            }
            // Conv2D would accept a tile that does not divide the image and quietly drop the
            // remainder, which is right for a convolution and a trap for a patch projection:
            // a 5-wide tile on 28 rows covers 25 of them and the other 3 are never seen.
            if (imageHeight % tile != 0 || imageWidth % tile != 0) {
                throw new IllegalArgumentException("a tile of " + tile + " does not divide a " + imageHeight + "x"
                        + imageWidth + " image, so part of it would never be seen");
            }
            if (dModel % heads != 0) {
                throw new IllegalArgumentException(heads + " heads do not divide " + dModel + " features");
            }
            return new TransformerEncoder(this);
        }

        private static void require(boolean given, String what) {
            if (!given) {
                throw new IllegalStateException(what + " was not set on this builder");
            }
        }
    }
}
