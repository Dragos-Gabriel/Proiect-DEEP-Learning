package com.handwriting.gan;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.util.Random;

public class GANModel {
    private final MultiLayerNetwork generator;
    private final MultiLayerNetwork discriminator;
    private final int latentDim = 100;
    private final Random rand = new Random();

    public GANModel() {
        this.generator = Generator.build(latentDim);
        this.discriminator = Discriminator.build();
    }

    public void train(DataSetIterator data, int epochs) throws Exception {
        int batchCount = 0;

        while (data.hasNext() && batchCount < epochs) {
            DataSet batch = data.next();
            INDArray realImages = batch.getFeatures();
            int batchSize = (int) realImages.size(0);

            // === 1. Train discriminator ===
            INDArray realLabels = Nd4j.ones(batchSize, 1);
            INDArray fakeLabels = Nd4j.zeros(batchSize, 1);

            INDArray noise = Nd4j.randn(batchSize, latentDim);
            INDArray fakeImages = generator.output(noise);

            discriminator.fit(realImages, realLabels);
            discriminator.fit(fakeImages, fakeLabels);

            // === 2. Train generator (manual feedback)
            INDArray newNoise = Nd4j.randn(batchSize, latentDim);
            INDArray generatedImages = generator.output(newNoise);

            // Obținem predicția discriminatorului pentru imaginile false
            INDArray dPred = discriminator.output(generatedImages);

            // Creăm o „etichetă țintă” care să fie 1 (adică real)
            INDArray targetOnes = Nd4j.onesLike(dPred);

            // Calculăm eroarea (pierdere simplificată)
            INDArray error = targetOnes.sub(dPred);
            // Actualizăm manual greutățile generatorului proporțional cu eroarea
            generator.params().subi(error.mean(0).mul(0.01));

            if (batchCount % 5 == 0) {
                System.out.println("Epoch " + batchCount + " - imagine generată salvată...");
                saveGeneratedImages(batchCount);
            }

            batchCount++;
        }
    }


    private void saveGeneratedImages(int epoch) throws Exception {
        File outDir = new File("output");
        if (!outDir.exists()) outDir.mkdirs();

        INDArray noise = Nd4j.randn(16, latentDim);
        INDArray samples = generator.output(noise);

        LatentSpaceVisualizer.saveAsImageGrid(samples, 4, 4,
                "output/generated_epoch_" + epoch + ".png");
    }
}

