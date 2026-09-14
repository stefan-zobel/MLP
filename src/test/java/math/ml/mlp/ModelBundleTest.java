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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ModelBundleTest {

    @TempDir
    Path dir;

    @Test
    void whatIsWrittenComesBackUnderTheSameKey() throws IOException {
        Path file = dir.resolve("m.zip");
        try (ModelBundle.Writer out = ModelBundle.write(file, 7)) {
            put(out, "a/weights", 1, 2, 3);
            put(out, "a/biases", 4);
            put(out, "b/gamma", 5, 6);
            out.finish();
        }

        try (ModelBundle.Reader in = ModelBundle.read(file)) {
            // deliberately not the order they were written in
            assertArrayEquals(new byte[] { 5, 6 }, get(in, "b/gamma"));
            assertArrayEquals(new byte[] { 1, 2, 3 }, get(in, "a/weights"));
            assertArrayEquals(new byte[] { 4 }, get(in, "a/biases"));
            assertEquals(7, in.step());
            in.requireFullyRead();
        }
    }

    @Test
    void nothingAppearsUntilTheBundleIsFinished() throws IOException {
        Path file = dir.resolve("m.zip");
        try (ModelBundle.Writer out = ModelBundle.write(file, 1)) {
            put(out, "a/weights", 1, 2, 3);
            assertFalse(Files.exists(file), "the target must stay untouched until finish()");
        }

        assertFalse(Files.exists(file), "a bundle that was never finished must not appear");
        assertFalse(Files.exists(dir.resolve("m.zip.tmp")), "and it must not leave its temporary file");
    }

    @Test
    void anOlderBundleIsReplacedWhole() throws IOException {
        Path file = dir.resolve("m.zip");
        try (ModelBundle.Writer out = ModelBundle.write(file, 1)) {
            put(out, "a/weights", 1);
            put(out, "gone/positions", 9);
            out.finish();
        }
        try (ModelBundle.Writer out = ModelBundle.write(file, 2)) {
            put(out, "a/weights", 2);
            out.finish();
        }

        try (ModelBundle.Reader in = ModelBundle.read(file)) {
            assertArrayEquals(new byte[] { 2 }, get(in, "a/weights"));
            assertEquals(2, in.step());
            assertThrows(IllegalStateException.class, () -> in.open("gone/positions"),
                    "entries of the older bundle must not survive");
        }
    }

    @Test
    void aDuplicateKeyIsRefused() throws IOException {
        Path file = dir.resolve("m.zip");
        try (ModelBundle.Writer out = ModelBundle.write(file, 1)) {
            put(out, "same/weights", 1);
            assertTrue(assertThrows(IllegalStateException.class, () -> put(out, "same/weights", 2)).getMessage()
                    .contains("same/weights"));
        }
    }

    @Test
    void theManifestIsNotAvailableAsAKey() throws IOException {
        Path file = dir.resolve("m.zip");
        try (ModelBundle.Writer out = ModelBundle.write(file, 1)) {
            assertThrows(IllegalArgumentException.class, () -> put(out, "manifest", 1));
        }
    }

    @Test
    void twoEntriesCannotBeOpenAtOnce() throws IOException {
        Path file = dir.resolve("m.zip");
        try (ModelBundle.Writer out = ModelBundle.write(file, 1)) {
            OutputStream first = out.open("a/weights");
            assertThrows(IllegalStateException.class, () -> out.open("b/weights"));
            first.close();
            out.open("b/weights").close();
        }
    }

    @Test
    void aMissingEntryIsReportedWithItsKey() throws IOException {
        Path file = dir.resolve("m.zip");
        try (ModelBundle.Writer out = ModelBundle.write(file, 1)) {
            put(out, "a/weights", 1);
            out.finish();
        }

        try (ModelBundle.Reader in = ModelBundle.read(file)) {
            assertTrue(assertThrows(IllegalStateException.class, () -> in.open("a/biases")).getMessage()
                    .contains("a/biases"));
        }
    }

    @Test
    void entriesNobodyAskedForAreRefused() throws IOException {
        Path file = dir.resolve("m.zip");
        try (ModelBundle.Writer out = ModelBundle.write(file, 1)) {
            put(out, "a/weights", 1);
            put(out, "b/weights", 2);
            out.finish();
        }

        try (ModelBundle.Reader in = ModelBundle.read(file)) {
            get(in, "a/weights");
            assertTrue(assertThrows(IllegalStateException.class, in::requireFullyRead).getMessage()
                    .contains("b/weights"));
        }
    }

    @Test
    void aMissingBundleIsReportedWithItsPath() {
        Path file = dir.resolve("absent.zip");
        assertTrue(assertThrows(IllegalStateException.class, () -> ModelBundle.read(file)).getMessage()
                .contains("absent.zip"));
    }

    @Test
    void aFileThatIsNotAZipIsRefused() throws IOException {
        Path file = dir.resolve("m.zip");
        Files.write(file, new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 });

        assertThrows(IllegalStateException.class, () -> ModelBundle.read(file));
    }

    @Test
    void aZipWithoutAManifestIsRefused() throws IOException {
        Path file = dir.resolve("m.zip");
        try (ZipOutputStream zip = new ZipOutputStream(Files.newOutputStream(file))) {
            zip.putNextEntry(new ZipEntry("a/weights"));
            zip.write(new byte[] { 1 });
            zip.closeEntry();
        }

        assertTrue(assertThrows(IllegalStateException.class, () -> ModelBundle.read(file)).getMessage()
                .contains("manifest"));
    }

    @Test
    void aManifestThatMiscountsItsEntriesIsRefused() throws IOException {
        Path file = dir.resolve("m.zip");
        try (ZipOutputStream zip = new ZipOutputStream(Files.newOutputStream(file))) {
            zip.putNextEntry(new ZipEntry("a/weights"));
            zip.write(new byte[] { 1 });
            zip.closeEntry();
            zip.putNextEntry(new ZipEntry("manifest"));
            zip.write("format=1\nstep=3\nentries=5\n".getBytes("ISO-8859-1"));
            zip.closeEntry();
        }

        assertTrue(assertThrows(IllegalStateException.class, () -> ModelBundle.read(file)).getMessage()
                .contains("5"));
    }

    @Test
    void aBundleOfAnotherFormatVersionIsRefused() throws IOException {
        Path file = dir.resolve("m.zip");
        try (ZipOutputStream zip = new ZipOutputStream(Files.newOutputStream(file))) {
            zip.putNextEntry(new ZipEntry("manifest"));
            zip.write("format=2\nstep=3\nentries=0\n".getBytes("ISO-8859-1"));
            zip.closeEntry();
        }

        assertTrue(assertThrows(IllegalStateException.class, () -> ModelBundle.read(file)).getMessage()
                .contains("format 2"));
    }

    @Test
    void theStepIsCarriedByTheBundleAndNotByAnEntry() throws IOException {
        Path file = dir.resolve("m.zip");
        try (ModelBundle.Writer out = ModelBundle.write(file, 1128)) {
            put(out, "a/weights", 1);
            out.finish();
        }

        try (ModelBundle.Reader in = ModelBundle.read(file)) {
            assertEquals(1128, in.step());
        }
    }

    @Test
    void aBundleOfManyEntriesSurvivesTheRoundTrip() throws IOException {
        Path file = dir.resolve("m.zip");
        int count = 200;
        try (ModelBundle.Writer out = ModelBundle.write(file, 5)) {
            for (int i = 0; i < count; ++i) {
                put(out, "layer" + i + "/weights", (byte) i, (byte) (i + 1));
            }
            out.finish();
        }

        try (ModelBundle.Reader in = ModelBundle.read(file)) {
            for (int i = 0; i < count; ++i) {
                assertArrayEquals(new byte[] { (byte) i, (byte) (i + 1) }, get(in, "layer" + i + "/weights"));
            }
            in.requireFullyRead();
        }
    }

    private static void put(ParameterSink sink, String key, int... bytes) throws IOException {
        try (OutputStream os = sink.open(key)) {
            for (int b : bytes) {
                os.write(b);
            }
        }
    }

    private static byte[] get(ParameterSource source, String key) throws IOException {
        try (InputStream is = source.open(key)) {
            return is.readAllBytes();
        }
    }
}
