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

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixF;

/**
 * The IDX file format shared by MNIST and its relatives.
 */
final class Idx {

    // one image per column, raw byte values; the dimensions come from the header, so this
    // reads any idx3 file and not only the 28 x 28 ones
    static MatrixF readImages(String path) throws IOException {
        try (DataInputStream ds = getDataInputStream(path)) {
            int imageCount = ds.readInt();
            int rowPixelCount = ds.readInt();
            int colPixelCount = ds.readInt();
            final int matrixRowCount = rowPixelCount * colPixelCount;
            // we store each image in a column of the returned matrix
            MatrixF images = Matrices.createF(matrixRowCount, imageCount);
            // images in the LeCun files are stored in row-major, so we store them line by
            // line into our column
            for (int col = 0; col < imageCount; ++col) {
                for (int row = 0; row < matrixRowCount; ++row) {
                    images.set(row, col, ds.readUnsignedByte());
                }
            }
            return images;
        }
    }

    // one one-hot column per label; the templates are built per call rather than cached
    // because the number of classes belongs to the dataset and not to this class
    static MatrixF readLabels(String path, int distinctLabels) throws IOException {
        MatrixF[] oneHot = new MatrixF[distinctLabels];
        for (int i = 0; i < distinctLabels; ++i) {
            oneHot[i] = Matrices.createF(distinctLabels, 1);
            oneHot[i].set(i, 0, 1.0f);
        }
        try (DataInputStream ds = getDataInputStream(path)) {
            int labelCount = ds.readInt();
            MatrixF labels = Matrices.createF(distinctLabels, labelCount);
            for (int i = 0; i < labelCount; ++i) {
                int label = ds.readUnsignedByte();
                if (label >= distinctLabels) {
                    // a dataset read with the wrong number of classes would otherwise fail
                    // somewhere further on, or not at all
                    throw new IOException(
                            path + " holds label " + label + " but was read as " + distinctLabels + " classes");
                }
                labels.setColumnInplace(i, oneHot[label]);
            }
            return labels;
        }
    }

    // package-private because MNIST, EMNIST and MNISTAugmenter all read these files
    static DataInputStream getDataInputStream(String path) throws IOException {
        DataInputStream ds = new DataInputStream(new BufferedInputStream(new FileInputStream(path)));
        // throw away magic number
        ds.readInt();
        return ds;
    }

    private Idx() {
        throw new AssertionError();
    }
}
