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
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

/**
 * Multi-head self attention over a {@code dModel x (batch * seqLen)} layout, where one
 * sample is a run of {@code seqLen} consecutive columns. The sequence length is fixed, which
 * is what lets the sequence axis fold into the column axis instead of needing a third one.
 *
 * <p>Every head owns its own projections, so a head is a whole matrix rather than a range of
 * rows: rows are not contiguous in a column-major layout, and slicing them would copy. The
 * projections are therefore whole-batch matrix products, while the two products inside a
 * head run per sample as plain loops, which also makes them independent of the alignment
 * effects an {@code sgemm} has.
 */
public class Attention extends AbstractLayer {

    /** Query projections, one per head, each dHead x dModel. */
    protected final Parameter[] queries;
    /** Key projections, one per head, each dHead x dModel. */
    protected final Parameter[] keys;
    /** Value projections, one per head, each dHead x dModel. */
    protected final Parameter[] values;
    /** Output projections, one per head, each dModel x dHead, summed into the result. */
    protected final Parameter[] outputs;
    /** Names the bundle entries of this layer; without a name it has none. */
    protected final String name;

    private final int dModel;
    private final int seqLen;
    private final int heads;
    private final int dHead;
    private final float scale;

    private MatrixF[] q;
    private MatrixF[] k;
    private MatrixF[] v;
    private MatrixF[] context;
    private MatrixF[] contextGrads;
    private MatrixF[] qGrads;
    private MatrixF[] kGrads;
    private MatrixF[] vGrads;
    /** The attention weights per head, kept for the backward pass. */
    private float[][] probabilities;
    /** Scratch for one pass over the scores; heads run one after the other, so one suffices. */
    private float[] scores;
    private MatrixF output;
    private MatrixF inputGrads;
    /** The batch the buffers above were sized for, or -1 before the first pass. */
    private int batch = -1;

    /**
     * Creates an attention layer with Glorot initialization from an unseeded draw.
     *
     * @param dModel features per token
     * @param seqLen tokens per sample
     * @param heads  number of attention heads; must divide {@code dModel}
     * @param name   names the bundle entries of this layer
     */
    public Attention(int dModel, int seqLen, int heads, String name) {
        this(dModel, seqLen, heads, name, Init.GLOROT, ThreadLocalRandom.current().nextLong());
    }

    /**
     * Creates an attention layer whose initialization is reproducible.
     *
     * @param dModel features per token
     * @param seqLen tokens per sample
     * @param heads  number of attention heads; must divide {@code dModel}
     * @param name   names the bundle entries of this layer
     * @param seed   seed from which every projection draws
     */
    public Attention(int dModel, int seqLen, int heads, String name, long seed) {
        this(dModel, seqLen, heads, name, Init.GLOROT, seed);
    }

    /**
     * The overloads without an {@code init} use {@link Init#GLOROT}.
     *
     * @param dModel features per token
     * @param seqLen tokens per sample
     * @param heads  number of attention heads; must divide {@code dModel}
     * @param name   names the bundle entries of this layer
     * @param init   the weight initialization scheme
     * @param seed   seed from which every projection draws
     */
    public Attention(int dModel, int seqLen, int heads, String name, Init init, long seed) {
        if (dModel <= 0 || seqLen <= 0 || heads <= 0) {
            throw new IllegalArgumentException(
                    "dModel " + dModel + ", seqLen " + seqLen + " and heads " + heads + " must all be positive");
        }
        if (dModel % heads != 0) {
            throw new IllegalArgumentException(heads + " heads do not divide " + dModel + " features");
        }
        this.dModel = dModel;
        this.seqLen = seqLen;
        this.heads = heads;
        this.dHead = dModel / heads;
        this.scale = 1.0f / (float) Math.sqrt(dHead);
        this.name = name;
        this.queries = new Parameter[heads];
        this.keys = new Parameter[heads];
        this.values = new Parameter[heads];
        this.outputs = new Parameter[heads];
        // one seed in, every projection drawn from it, so the whole layer is reproducible
        SplittableRandom rnd = new SplittableRandom(seed);
        float inBound = init.bound(dModel, dHead);
        float outBound = init.bound(dHead, dModel);
        for (int h = 0; h < heads; ++h) {
            queries[h] = projection("q" + h, dHead, dModel, inBound, rnd.nextLong());
            keys[h] = projection("k" + h, dHead, dModel, inBound, rnd.nextLong());
            values[h] = projection("v" + h, dHead, dModel, inBound, rnd.nextLong());
            outputs[h] = projection("o" + h, dModel, dHead, outBound, rnd.nextLong());
        }
    }

    private static Parameter projection(String suffix, int rows, int cols, float bound, long seed) {
        return new Parameter(suffix, Matrices.randomUniformF(rows, cols, -bound, bound, seed), true);
    }

    /** Every projection of every head, in the order query, key, value, output. */
    @Override
    public List<Parameter> parameters() {
        List<Parameter> all = new ArrayList<>(4 * heads);
        for (int h = 0; h < heads; ++h) {
            all.add(queries[h]);
            all.add(keys[h]);
            all.add(values[h]);
            all.add(outputs[h]);
        }
        return List.copyOf(all);
    }

    /** Tokens per sample, as declared. */
    public int sequenceLength() {
        return seqLen;
    }

    /** Features per head. */
    public int headWidth() {
        return dHead;
    }

    @Override
    public MatrixF forward(MatrixF in) {
        if (in.numRows() != dModel) {
            throw new IllegalArgumentException("expected " + dModel + " features, got " + in.numRows());
        }
        int cols = in.numColumns();
        if (cols % seqLen != 0) {
            throw new IllegalArgumentException(cols + " columns is not a whole number of sequences of " + seqLen);
        }
        int m = cols / seqLen;
        ensureBuffers(m);
        super.forward(in);
        for (int h = 0; h < heads; ++h) {
            queries[h].value().mult(in, q[h]);
            keys[h].value().mult(in, k[h]);
            values[h].value().mult(in, v[h]);
        }
        for (int h = 0; h < heads; ++h) {
            int head = h;
            // one sample owns a contiguous run of columns and of score entries, so the
            // tasks are disjoint and each of them keeps the order the flat loop had
            IntStream.range(0, m).parallel().forEach(s -> attend(head, s));
        }
        outputs[0].value().mult(context[0], output);
        for (int h = 1; h < heads; ++h) {
            outputs[h].value().multAdd(context[h], output);
        }
        return output;
    }

    private void attend(int head, int s) {
        float[] qa = q[head].getArrayUnsafe();
        float[] ka = k[head].getArrayUnsafe();
        float[] va = v[head].getArrayUnsafe();
        float[] pa = probabilities[head];
        float[] ca = context[head].getArrayUnsafe();
        int block = s * seqLen * seqLen;
        int col0 = s * seqLen * dHead;
        for (int j = 0; j < seqLen; ++j) {
            int qo = col0 + j * dHead;
            int so = block + j * seqLen;
            for (int i = 0; i < seqLen; ++i) {
                int ko = col0 + i * dHead;
                float dot = 0.0f;
                for (int d = 0; d < dHead; ++d) {
                    dot += qa[qo + d] * ka[ko + d];
                }
                scores[so + i] = dot * scale;
            }
            // the same routine the Softmax layer uses, so the overflow handling is shared
            math.dl.Softmax.softmaxF(seqLen, so, scores, so, pa);
        }
        Arrays.fill(ca, col0, col0 + seqLen * dHead, 0.0f);
        for (int j = 0; j < seqLen; ++j) {
            int co = col0 + j * dHead;
            int so = block + j * seqLen;
            for (int i = 0; i < seqLen; ++i) {
                float p = pa[so + i];
                int vo = col0 + i * dHead;
                for (int d = 0; d < dHead; ++d) {
                    ca[co + d] += va[vo + d] * p;
                }
            }
        }
    }

    @Override
    public MatrixF backward(MatrixF outputGrads) {
        if (mode == NetworkMode.INFER) {
            return null;
        }
        int m = outputGrads.numColumns() / seqLen;
        for (int h = 0; h < heads; ++h) {
            // the projections are shared by every token, so the mean over the batch divides
            // by the number of samples and not by the number of columns
            outputGrads.transBmult(context[h], outputs[h].grad()).scaleInplace(1.0f / m);
            outputs[h].value().transAmult(outputGrads, contextGrads[h]);
        }
        for (int h = 0; h < heads; ++h) {
            int head = h;
            IntStream.range(0, m).parallel().forEach(s -> attendBackward(head, s));
        }
        for (int h = 0; h < heads; ++h) {
            qGrads[h].transBmult(input, queries[h].grad()).scaleInplace(1.0f / m);
            kGrads[h].transBmult(input, keys[h].grad()).scaleInplace(1.0f / m);
            vGrads[h].transBmult(input, values[h].grad()).scaleInplace(1.0f / m);
        }
        queries[0].value().transAmult(qGrads[0], inputGrads);
        keys[0].value().transAmultAdd(kGrads[0], inputGrads);
        values[0].value().transAmultAdd(vGrads[0], inputGrads);
        for (int h = 1; h < heads; ++h) {
            queries[h].value().transAmultAdd(qGrads[h], inputGrads);
            keys[h].value().transAmultAdd(kGrads[h], inputGrads);
            values[h].value().transAmultAdd(vGrads[h], inputGrads);
        }
        input = null;
        return inputGrads;
    }

    private void attendBackward(int head, int s) {
        float[] qa = q[head].getArrayUnsafe();
        float[] ka = k[head].getArrayUnsafe();
        float[] va = v[head].getArrayUnsafe();
        float[] pa = probabilities[head];
        float[] dc = contextGrads[head].getArrayUnsafe();
        float[] dq = qGrads[head].getArrayUnsafe();
        float[] dk = kGrads[head].getArrayUnsafe();
        float[] dv = vGrads[head].getArrayUnsafe();
        int block = s * seqLen * seqLen;
        int col0 = s * seqLen * dHead;
        Arrays.fill(dv, col0, col0 + seqLen * dHead, 0.0f);
        for (int j = 0; j < seqLen; ++j) {
            int co = col0 + j * dHead;
            int so = block + j * seqLen;
            for (int i = 0; i < seqLen; ++i) {
                int vo = col0 + i * dHead;
                float p = pa[so + i];
                float dot = 0.0f;
                for (int d = 0; d < dHead; ++d) {
                    dv[vo + d] += dc[co + d] * p;
                    dot += va[vo + d] * dc[co + d];
                }
                scores[so + i] = dot;
            }
            // the softmax of one query column, in the same closed form the Softmax layer
            // uses, and then the scale the scores were multiplied by on the way in
            double sum = 0.0;
            for (int i = 0; i < seqLen; ++i) {
                sum += (double) scores[so + i] * pa[so + i];
            }
            float shift = (float) sum;
            for (int i = 0; i < seqLen; ++i) {
                scores[so + i] = pa[so + i] * (scores[so + i] - shift) * scale;
            }
        }
        Arrays.fill(dq, col0, col0 + seqLen * dHead, 0.0f);
        Arrays.fill(dk, col0, col0 + seqLen * dHead, 0.0f);
        for (int j = 0; j < seqLen; ++j) {
            int qo = col0 + j * dHead;
            int so = block + j * seqLen;
            for (int i = 0; i < seqLen; ++i) {
                int ko = col0 + i * dHead;
                float ds = scores[so + i];
                for (int d = 0; d < dHead; ++d) {
                    dq[qo + d] += ka[ko + d] * ds;
                    dk[ko + d] += qa[qo + d] * ds;
                }
            }
        }
    }

    private void ensureBuffers(int m) {
        if (batch == m) {
            return;
        }
        int cols = m * seqLen;
        q = new MatrixF[heads];
        k = new MatrixF[heads];
        v = new MatrixF[heads];
        context = new MatrixF[heads];
        contextGrads = new MatrixF[heads];
        qGrads = new MatrixF[heads];
        kGrads = new MatrixF[heads];
        vGrads = new MatrixF[heads];
        probabilities = new float[heads][];
        for (int h = 0; h < heads; ++h) {
            q[h] = Matrices.createF(dHead, cols);
            k[h] = Matrices.createF(dHead, cols);
            v[h] = Matrices.createF(dHead, cols);
            context[h] = Matrices.createF(dHead, cols);
            contextGrads[h] = Matrices.createF(dHead, cols);
            qGrads[h] = Matrices.createF(dHead, cols);
            kGrads[h] = Matrices.createF(dHead, cols);
            vGrads[h] = Matrices.createF(dHead, cols);
            probabilities[h] = new float[m * seqLen * seqLen];
        }
        scores = new float[m * seqLen * seqLen];
        output = Matrices.createF(dModel, cols);
        inputGrads = Matrices.createF(dModel, cols);
        batch = m;
    }
    @Override
    public void writeParameters(ParameterSink sink) throws IOException {
        ParameterStore.requireName(name, this);
        // every projection of every head, named as parameters() names it, so that the
        // two directions cannot drift apart
        for (Parameter p : parameters()) {
            ParameterStore.write(sink, name + "/" + p.name(), p.value());
        }
    }

    @Override
    public void readParameters(ParameterSource source) throws IOException {
        ParameterStore.requireName(name, this);
        for (Parameter p : parameters()) {
            ParameterStore.read(source, name + "/" + p.name(), p.value());
        }
    }
}
