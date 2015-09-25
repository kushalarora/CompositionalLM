package com.kushalarora.compositionalLM.utils;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.plot.Tsne;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.util.List;

/**
 * Created by arorak on 9/23/15.
 */

@Slf4j
public class Visualization {

    public static void saveTNSEVisualization(String filename, INDArray X, List<String> wordList) throws IOException {

        log.info("Build tnse model....");
        Tsne tsne = new Tsne.Builder()
                .setMaxIter(1000)
                .normalize(true)
                .learningRate(500)
                .useAdaGrad(false)
                .usePca(false)
                .build();

        log.info("Store TSNE Coordinates for Plotting in file: {}", filename);
        tsne.plot(X, 2, wordList, filename);
    }
}
