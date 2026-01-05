package com.handwriting.gan;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class ConditionalGAN extends GANModel {

    public INDArray generateDigit(int digit) {
        int latentDim = 100;
        INDArray noise = Nd4j.randn(1, latentDim);
        INDArray label = Nd4j.zeros(1, 10);
        label.putScalar(0, digit, 1);
        INDArray input = Nd4j.hstack(noise, label);
        return input;
    }
}

