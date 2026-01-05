package com.handwriting.gan;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.HeatMapChart;
import org.knowm.xchart.style.HeatMapStyler;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;
import java.util.ArrayList;
import java.util.List;

public class LatentSpaceVisualizer {

    public static void saveAsImageGrid(INDArray samples, int rows, int cols, String path) throws Exception {
        int imgH = 28, imgW = 28;
        BufferedImage out = new BufferedImage(cols * imgW, rows * imgH, BufferedImage.TYPE_BYTE_GRAY);
        int idx = 0;
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) {
                if (idx >= samples.rows()) break;
                INDArray img = samples.getRow(idx++).reshape(28, 28);
                for (int y = 0; y < imgH; y++)
                    for (int x = 0; x < imgW; x++) {
                        int val = (int) ((img.getDouble(y, x) + 1) * 127.5);
                        int rgb = new Color(val, val, val).getRGB();
                        out.setRGB(c * imgW + x, r * imgH + y, rgb);
                    }
            }
        ImageIO.write(out, "png", new File(path));
    }

    // Vizualizare latent 2D
    public static void visualize2D(INDArray z, INDArray labels, String path) throws Exception {
        int n = (int) z.size(0);

        List<Double> xs = new ArrayList<>();
        List<Double> ys = new ArrayList<>();
        List<Integer> lbls = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            xs.add(z.getDouble(i, 0));
            ys.add(z.getDouble(i, 1));
            lbls.add(Nd4j.argMax(labels.getRow(i)).getInt(0));  // ✅ corect
        }

        org.knowm.xchart.XYChart chart = new org.knowm.xchart.XYChartBuilder()
                .width(600)
                .height(500)
                .title("Latent Space Visualization")
                .xAxisTitle("Z1")
                .yAxisTitle("Z2")
                .build();

        for (int digit = 0; digit < 10; digit++) {
            List<Double> dx = new ArrayList<>();
            List<Double> dy = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (lbls.get(i) == digit) {
                    dx.add(xs.get(i));
                    dy.add(ys.get(i));
                }
            }
            chart.addSeries("Digit " + digit, dx, dy);
        }

        BitmapEncoder.saveBitmap(chart, path, BitmapEncoder.BitmapFormat.PNG);
        System.out.println("Harta latentă salvată la: " + path);
    }


}

