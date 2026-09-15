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

import java.io.BufferedOutputStream;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Properties;
import java.util.Set;
import java.util.zip.Deflater;
import java.util.zip.ZipEntry;
import java.util.zip.ZipException;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

/**
 * A whole network in one zip file: one entry per matrix, named
 * {@code layerName/parameter}, plus a manifest carrying the format version and
 * the step the parameters were written at.
 *
 * <p>One step for the whole bundle is the point of the format: weights and
 * optimizer moments cannot come from two different moments, because there is
 * one write.
 */
public final class ModelBundle {

    /** The entry holding the manifest, which is not a parameter. */
    static final String MANIFEST = "manifest";

    private static final int FORMAT_VERSION = 1;
    private static final String FORMAT = "format";
    private static final String STEP = "step";
    private static final String WRITTEN = "written";
    private static final String ENTRIES = "entries";

    private ModelBundle() {
        throw new AssertionError();
    }

    /**
     * Opens a bundle for writing. Nothing appears at {@code file} until
     * {@link Writer#finish()} runs.
     *
     * @param file where the finished bundle goes
     * @param step the training step every parameter in it comes from
     * @return the open bundle
     * @throws IOException if the temporary file cannot be created
     */
    public static Writer write(Path file, int step) throws IOException {
        return new Writer(file, step);
    }

    /**
     * Opens an existing bundle for reading.
     *
     * @param file the bundle
     * @return the open bundle
     * @throws IOException           if it cannot be read
     * @throws IllegalStateException if it is missing or is not a bundle
     */
    public static Reader read(Path file) throws IOException {
        return new Reader(file);
    }

    /**
     * A bundle being written. Write every entry, then call {@link #finish()};
     * closing without it discards the whole bundle.
     */
    public static final class Writer implements ParameterSink, Closeable {

        private final Path target;
        private final Path temp;
        private final ZipOutputStream zip;
        private final Set<String> keys = new LinkedHashSet<>();
        private final int step;
        private boolean entryOpen;
        private boolean finished;
        private boolean closed;

        private Writer(Path file, int step) throws IOException {
            this.target = file;
            this.temp = file.resolveSibling(file.getFileName() + ".tmp");
            this.step = step;
            Path dir = file.toAbsolutePath().getParent();
            if (dir != null) {
                Files.createDirectories(dir);
            }
            zip = new ZipOutputStream(new BufferedOutputStream(Files.newOutputStream(temp)));
            // the payload is trained floats, which deflate cannot shrink; the zip is here for
            // the structure and not for the size
            zip.setLevel(Deflater.NO_COMPRESSION);
        }

        @Override
        public OutputStream open(String key) throws IOException {
            if (closed) {
                throw new IllegalStateException("this bundle is closed");
            }
            if (entryOpen) {
                throw new IllegalStateException("the entry before " + key + " was not closed");
            }
            if (MANIFEST.equals(key)) {
                throw new IllegalArgumentException(MANIFEST + " is the name of the manifest");
            }
            if (!keys.add(key)) {
                // two layers of the same name would otherwise silently overwrite each other,
                // which is what the one file per matrix layout did
                throw new IllegalStateException(key + " was written before; two layers cannot share a name");
            }
            zip.putNextEntry(new ZipEntry(key));
            entryOpen = true;
            return new EntryStream();
        }

        /**
         * How many entries have been written so far, so that a caller can tell what one
         * layer contributed.
         *
         * @return the number of entries in this bundle
         */
        public int entryCount() {
            return keys.size();
        }

        /**
         * Writes the manifest and moves the finished bundle into place, which is
         * the only moment anything is visible at the target path.
         *
         * @throws IOException if writing or the move fails
         */
        public void finish() throws IOException {
            if (finished) {
                throw new IllegalStateException("this bundle is already finished");
            }
            if (entryOpen) {
                throw new IllegalStateException("an entry is still open");
            }
            Properties manifest = new Properties();
            manifest.setProperty(FORMAT, Integer.toString(FORMAT_VERSION));
            manifest.setProperty(STEP, Integer.toString(step));
            manifest.setProperty(WRITTEN, Instant.now().toString());
            // last, so that it can count the entries it describes
            manifest.setProperty(ENTRIES, Integer.toString(keys.size()));
            zip.putNextEntry(new ZipEntry(MANIFEST));
            manifest.store(zip, "network parameters");
            zip.closeEntry();
            zip.close();
            Files.move(temp, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            // after the move, so that a move that fails still counts as unfinished and close()
            // takes the temporary file away
            finished = true;
        }

        /** Discards an unfinished bundle, so that a failed write leaves nothing behind. */
        @Override
        public void close() throws IOException {
            if (closed) {
                return;
            }
            closed = true;
            if (!finished) {
                zip.close();
                Files.deleteIfExists(temp);
            }
        }

        /** Ends one entry on close instead of closing the zip it belongs to. */
        private final class EntryStream extends OutputStream {

            @Override
            public void write(int b) throws IOException {
                zip.write(b);
            }

            @Override
            public void write(byte[] b, int off, int len) throws IOException {
                zip.write(b, off, len);
            }

            @Override
            public void close() throws IOException {
                if (entryOpen) {
                    entryOpen = false;
                    zip.closeEntry();
                }
            }
        }
    }

    /** A bundle being read. Entries are addressed by name, so their order does not matter. */
    public static final class Reader implements ParameterSource, Closeable {

        private final Path file;
        private final ZipFile zip;
        private final Set<String> read = new HashSet<>();
        private final int step;

        private Reader(Path file) throws IOException {
            this.file = file;
            if (!Files.isReadable(file)) {
                throw new IllegalStateException("no model bundle at " + file);
            }
            ZipFile opened;
            try {
                opened = new ZipFile(file.toFile());
            } catch (ZipException e) {
                throw new IllegalStateException(file + " is not a model bundle", e);
            }
            this.zip = opened;
            ZipEntry entry = zip.getEntry(MANIFEST);
            if (entry == null) {
                zip.close();
                throw new IllegalStateException(file + " holds no " + MANIFEST + ", so it is not a model bundle");
            }
            Properties manifest = new Properties();
            try (InputStream is = zip.getInputStream(entry)) {
                manifest.load(is);
            }
            int format = number(manifest, FORMAT);
            if (format != FORMAT_VERSION) {
                zip.close();
                throw new IllegalStateException(file + " is format " + format + ", this is " + FORMAT_VERSION);
            }
            int entries = number(manifest, ENTRIES);
            int present = parameterEntries();
            if (entries != present) {
                zip.close();
                throw new IllegalStateException(
                        file + " describes " + entries + " parameter entries and holds " + present);
            }
            this.step = number(manifest, STEP);
        }

        /**
         * The training step every parameter in this bundle comes from.
         *
         * @return the step the bundle was written at
         */
        public int step() {
            return step;
        }

        @Override
        public InputStream open(String key) throws IOException {
            ZipEntry entry = zip.getEntry(key);
            if (entry == null) {
                throw new IllegalStateException(file + " holds no entry " + key);
            }
            read.add(key);
            return zip.getInputStream(entry);
        }

        /**
         * Refuses a bundle holding parameters nobody asked for, which is how a
         * network loaded from the bundle of a different architecture is caught.
         */
        public void requireFullyRead() {
            List<String> left = new ArrayList<>();
            for (Enumeration<? extends ZipEntry> e = zip.entries(); e.hasMoreElements();) {
                String name = e.nextElement().getName();
                if (!MANIFEST.equals(name) && !read.contains(name)) {
                    left.add(name);
                }
            }
            if (!left.isEmpty()) {
                throw new IllegalStateException(file + " holds " + left.size()
                        + " entries this network did not ask for, the first of them " + left.get(0));
            }
        }

        @Override
        public void close() throws IOException {
            zip.close();
        }

        private int parameterEntries() {
            int count = 0;
            for (Enumeration<? extends ZipEntry> e = zip.entries(); e.hasMoreElements();) {
                if (!MANIFEST.equals(e.nextElement().getName())) {
                    ++count;
                }
            }
            return count;
        }

        private int number(Properties manifest, String key) throws IOException {
            String value = manifest.getProperty(key);
            if (value == null) {
                zip.close();
                throw new IllegalStateException(file + " has no " + key + " in its " + MANIFEST);
            }
            try {
                return Integer.parseInt(value.trim());
            } catch (NumberFormatException e) {
                zip.close();
                throw new IllegalStateException(file + " has " + key + " = " + value + " in its " + MANIFEST, e);
            }
        }
    }
}
