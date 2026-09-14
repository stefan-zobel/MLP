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

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** Holds the parameter list, the step counter and the schedule of an optimizer. */
public abstract class AbstractOptimizer implements Optimizer {

    // "OPT1", so a file that is not one of these is refused instead of read as garbage
    private static final int MAGIC = 0x4F505431;
    private static final int FORMAT_VERSION = 1;

    /** The registered parameters, in registration order. */
    protected final List<Parameter> parameters = new ArrayList<>();

    private final LearningRateSchedule schedule;
    private int step;

    // infinity rather than zero is "off": zero is a threshold someone may legitimately
    // ask for, so it must not double as the sentinel
    private double maxGradNorm = Double.POSITIVE_INFINITY;
    private double lastNorm;
    private int clippedSteps;

    /** Names the state file, or null while this optimizer is not persisted at all. */
    private String stateName;
    private boolean storing;

    /**
     * For subclasses.
     *
     * @param schedule supplies the learning rate of every step
     */
    protected AbstractOptimizer(LearningRateSchedule schedule) {
        this.schedule = Objects.requireNonNull(schedule, "schedule");
    }

    @Override
    public final void add(Parameter p) {
        Objects.requireNonNull(p, "parameter");
        for (Parameter known : parameters) {
            // by identity: two parameters of the same shape and name are still two
            if (known == p) {
                throw new IllegalArgumentException("parameter is already registered: " + p);
            }
        }
        parameters.add(p);
        registered(p);
    }

    @Override
    public final void step() {
        // before anything else, so that update() always sees a gradient that already
        // satisfies the bound and both hot loops can stay as they are
        clip();
        float rate = schedule.rate(++step);
        beginStep(rate, step);
        for (int i = 0; i < parameters.size(); ++i) {
            update(parameters.get(i), i);
        }
    }

    /**
     * Scales all gradients down together whenever their global norm exceeds
     * {@code maxNorm}, from now on. Note that this bounds the gradient, not the step:
     * under momentum the velocity still reaches {@code 1 / (1 - momentum)} times the
     * clipped gradient, and the decoupled weight decay of {@link Adam} is unaffected
     * by design.
     *
     * @param maxNorm the largest global gradient norm to leave alone, positive;
     *                {@link Float#POSITIVE_INFINITY} switches clipping off again
     * @return this optimizer
     * @throws IllegalArgumentException if {@code maxNorm} is not positive
     */
    public final AbstractOptimizer clipGradientNorm(float maxNorm) {
        // a negative threshold would make the factor negative and turn every step into
        // gradient ascent, silently; NaN would disable clipping while still paying for
        // the norm; zero would erase every gradient including the non-finite ones this
        // deliberately lets through
        if (!(maxNorm > 0.0f)) {
            throw new IllegalArgumentException("maxNorm must be positive: " + maxNorm);
        }
        this.maxGradNorm = maxNorm;
        return this;
    }

    private void clip() {
        if (maxGradNorm == Double.POSITIVE_INFINITY) {
            return;
        }
        double norm = globalGradientNorm();
        lastNorm = norm;
        // A non-finite norm is left alone: every term is non-negative, so the sum is
        // infinite or NaN only because an element is, and scaling those away would hide
        // a divergence rather than report it. NaN also fails the comparison by itself.
        if (!(norm > maxGradNorm) || Double.isInfinite(norm)) {
            return;
        }
        ++clippedSteps;
        // the factor stays double: a finite norm can reach 1e43, where the float
        // reciprocal would be subnormal and lose most of its significand
        double factor = maxGradNorm / norm;
        for (int i = 0; i < parameters.size(); ++i) {
            float[] g = parameters.get(i).grad().getArrayUnsafe();
            for (int k = 0; k < g.length; ++k) {
                g[k] = (float) (g[k] * factor);
            }
        }
    }

    /**
     * Called once per parameter, when it is registered, for subclasses that keep
     * per-parameter state. The default does nothing.
     *
     * @param p the parameter that was just added
     */
    protected void registered(Parameter p) {
        // stateless by default
    }

    /**
     * Called once per step, before any parameter is updated.
     *
     * @param rate the learning rate the schedule returned
     * @param step the step number, counted from one
     */
    protected abstract void beginStep(float rate, int step);

    /**
     * Applies the update prepared by the preceding {@link #beginStep} to one parameter.
     *
     * @param p     the parameter to update
     * @param index its position in {@link #parameters}, for per-parameter state
     */
    protected abstract void update(Parameter p, int index);

    /**
     * The Euclidean norm of the gradients of every registered parameter, taken
     * together as one vector.
     *
     * @return the global gradient norm
     */
    public final float gradientNorm() {
        return (float) globalGradientNorm();
    }

    /**
     * The global gradient norm of the last step, as it was <em>before</em> clipping.
     *
     * @return the norm, or zero while clipping is switched off
     */
    public final float lastGradientNorm() {
        return (float) lastNorm;
    }

    /**
     * How many steps were scaled down because their gradients exceeded the bound.
     *
     * @return the number of clipped steps
     */
    public final int clippedSteps() {
        return clippedSteps;
    }

    // In double, and not for overflow: the square of the largest float is about 1.2e77,
    // so millions of them still sum comfortably. MatrixF.normF() would do the
    // overflow-resistant scaled algorithm instead, at a branch and two divides per
    // element, and it is per matrix while this norm spans all of them.
    private double globalGradientNorm() {
        double sum = 0.0;
        for (int i = 0; i < parameters.size(); ++i) {
            float[] g = parameters.get(i).grad().getArrayUnsafe();
            for (int k = 0; k < g.length; ++k) {
                sum += (double) g[k] * g[k];
            }
        }
        return Math.sqrt(sum);
    }

    /**
     * How many steps have been applied.
     *
     * @return the step counter
     */
    public final int steps() {
        return step;
    }

    // -------------------------------------------------------------------------
    // State that outlives the process
    // -------------------------------------------------------------------------

    /**
     * Names the file {@code o_<name>} this optimizer's state is read from and written to,
     * under the same two directories the layers use.
     *
     * @param name  identifies the state file
     * @param store let {@link #storeState()} write it, as a layer's flag does
     * @return this optimizer
     */
    public final AbstractOptimizer persistAs(String name, boolean store) {
        this.stateName = Objects.requireNonNull(name, "name");
        this.storing = store;
        return this;
    }

    /**
     * Writes the step counter and whatever per-parameter state the subclass keeps, if a name
     * and the store flag were given. Called by the network along with the layers, so that the
     * weights and this file cannot come from different steps.
     */
    @Override
    public final void storeState() {
        if (stateName == null || !storing) {
            return;
        }
        Path dir = Paths.get(ParameterStore.STORE_DIR);
        Path target = dir.resolve("o_" + stateName);
        Path tmp = dir.resolve("o_" + stateName + ".tmp");
        try {
            Files.createDirectories(dir);
            try (DataOutputStream out = new DataOutputStream(
                    new BufferedOutputStream(Files.newOutputStream(tmp)))) {
                out.writeInt(MAGIC);
                out.writeInt(FORMAT_VERSION);
                out.writeUTF(kind());
                out.writeInt(step);
                out.writeInt(clippedSteps);
                out.writeDouble(lastNorm);
                out.writeInt(parameters.size());
                for (Parameter p : parameters) {
                    out.writeInt(p.value().getArrayUnsafe().length);
                }
                writeState(out);
            }
            // so this one file is never half written; it does not make the set of layer
            // files atomic, which needs a format that holds a whole network
            Files.move(tmp, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Reads back what {@link #storeState()} wrote. Not on {@link Optimizer} and deliberately
     * not symmetric with its counterpart: the subclass state is sized from the registered
     * parameters, so this can only run once the network is complete, and an explicit call by
     * the program is the one place where that ordering is visible.
     *
     * @return the step the run continues at, for the caller to log
     * @throws IllegalStateException if there is no name, no file, or the file does not describe
     *                               this optimizer and these parameters
     */
    public final int loadState() {
        if (stateName == null) {
            throw new IllegalStateException("persistAs(...) has to name the state before it can be read");
        }
        Path source = Paths.get(ParameterStore.LOAD_DIR).resolve("o_" + stateName);
        if (!Files.isReadable(source)) {
            throw new IllegalStateException("no optimizer state at " + source);
        }
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(source)))) {
            if (in.readInt() != MAGIC) {
                throw new IllegalStateException(source + " is not an optimizer state");
            }
            int version = in.readInt();
            if (version != FORMAT_VERSION) {
                throw new IllegalStateException(source + " is format " + version + ", this is " + FORMAT_VERSION);
            }
            String written = in.readUTF();
            if (!kind().equals(written)) {
                throw new IllegalStateException(source + " was written by " + written + ", this is " + kind());
            }
            int loadedStep = in.readInt();
            int loadedClipped = in.readInt();
            double loadedNorm = in.readDouble();
            checkShapes(in, source);
            readState(in);
            step = loadedStep;
            clippedSteps = loadedClipped;
            lastNorm = loadedNorm;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return step;
    }

    // Every silent way to resume into the wrong network ends here: one parameter more, one
    // layer wider, the same layers registered in another order.
    private void checkShapes(DataInputStream in, Path source) throws IOException {
        int count = in.readInt();
        if (count != parameters.size()) {
            throw new IllegalStateException(source + " holds " + count + " parameters and this optimizer has "
                    + parameters.size() + "; loadState() belongs after the network is built");
        }
        for (int i = 0; i < count; ++i) {
            int length = in.readInt();
            int own = parameters.get(i).value().getArrayUnsafe().length;
            if (length != own) {
                throw new IllegalStateException(source + " holds " + length + " elements for parameter " + i
                        + " and this optimizer has " + own);
            }
        }
    }

    /**
     * Distinguishes the state of one optimizer class from another's.
     *
     * @return a stable name for this kind of optimizer
     */
    protected abstract String kind();

    /**
     * Writes the hyperparameters that must match on a resume, then the per-parameter state in
     * registration order.
     *
     * @param out the stream to write to
     * @throws IOException if writing fails
     */
    protected abstract void writeState(DataOutputStream out) throws IOException;

    /**
     * Reads back what {@link #writeState} wrote, rejecting hyperparameters that differ.
     *
     * @param in the stream to read from
     * @throws IOException if reading fails
     */
    protected abstract void readState(DataInputStream in) throws IOException;

    /**
     * Writes one per-parameter state array, for subclasses.
     *
     * @param out the stream to write to
     * @param a   the array
     * @throws IOException if writing fails
     */
    protected static void writeArray(DataOutputStream out, float[] a) throws IOException {
        for (float f : a) {
            out.writeFloat(f);
        }
    }

    /**
     * Fills one per-parameter state array, for subclasses. The length was checked before the
     * subclass part of the file was reached.
     *
     * @param in the stream to read from
     * @param a  the array to fill
     * @throws IOException if reading fails
     */
    protected static void readArray(DataInputStream in, float[] a) throws IOException {
        for (int i = 0; i < a.length; ++i) {
            a[i] = in.readFloat();
        }
    }

    /**
     * Refuses a hyperparameter that differs from the one the state was written with. Exact,
     * deliberately: both sides are constructor arguments and not computed values, and a moment
     * decayed at one beta and then read at another is wrong in a way that looks like slow
     * learning.
     *
     * @param written what the file holds
     * @param own     what this optimizer was built with
     * @param what    the name of the hyperparameter, for the message
     * @throws IllegalStateException if the two differ
     */
    protected static void expect(float written, float own, String what) {
        if (written != own) {
            throw new IllegalStateException("state was written with " + what + " " + written + ", this is " + own);
        }
    }
}
