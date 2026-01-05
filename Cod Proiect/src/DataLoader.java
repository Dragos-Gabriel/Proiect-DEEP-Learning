package com.handwriting.gan;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class DataLoader {
    public static DataSetIterator loadMNIST(int batchSize) throws Exception {
        return new MnistDataSetIterator(batchSize, true, 12345);
    }
}



