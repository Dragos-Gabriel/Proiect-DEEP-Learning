package com.handwriting.gan;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Utils {
    public static INDArray oneHotRange(int n, int numClasses) {
        INDArray out = Nd4j.zeros(n, numClasses);
        for (int i = 0; i < n; i++) out.putScalar(new int[]{i, i % numClasses}, 1);
        return out;
    }
}
