package com.kushalarora.compositionalLM.options;

import com.google.common.collect.Lists;
import com.kushalarora.compositionalLM.caching.CacheFactory;
import com.kushalarora.compositionalLM.optimizer.OptimizerFactory;
import lombok.ToString;
import org.apache.commons.configuration.Configuration;
import org.apache.commons.io.filefilter.WildcardFileFilter;

import java.io.File;
import java.io.FileFilter;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by karora on 6/14/15.
 */
@ToString
public class TrainOptions implements Serializable {
    public String[] trainFiles;
    public String[] validationFiles;
    public boolean validate;
    public int maxEpochs;
    public double tolerance;
    public int batchSize;
    public int validationFreq;
    public double learningRate;
    public boolean parallel;
    public int nThreads;
    public String cacheServer;
    public int cachePort;
    public CacheFactory.CacheType cacheType;
    public OptimizerFactory.OptimizerType optimizer;
    public int validBatchSize;
    public boolean saveVisualization;
    public String visualizationFilename;

    public TrainOptions(Configuration config) throws IOException {
        trainFiles =
                config.getStringArray("trainFiles");
        validate =
                config.getBoolean("validate", false);
        validationFiles =
                config.getStringArray("validationFiles");
        maxEpochs =
                config.getInt("maxEpochs", 10);
        tolerance =
                config.getDouble("tolerance", 1e-3);
        batchSize =
                config.getInt("batchSize", 100);
        validationFreq =
                config.getInt("validationFreq", 5);
        parallel =
                config.getBoolean("parallel", false);
        nThreads =
                config.getInt("nThreads", 0);
        cacheServer =
                config.getString("cacheServer", "localhost");

        cachePort =
                config.getInt("cachePort", 3030);

        cacheType =
                CacheFactory.CacheType.fromString(
                        config.getString("cacheType", "none"));
        learningRate =
                config.getDouble("learningRate", 0.0);

        optimizer =
                OptimizerFactory.OptimizerType.fromString(
                        config.getString("optimizerType", "sgd"));

        List<String> trainList = Lists.newArrayList(trainFiles);
        trainList.addAll(getFilesFromDir(
                config.getString("trainDir", null),
                config.getString("suffix", null)));

        trainFiles = trainList.toArray(trainFiles);

        List<String> validList = Lists.newArrayList(validationFiles);
        validList.addAll(getFilesFromDir(
                config.getString("validDir", null),
                config.getString("suffix", null)));


        validationFiles = validList.toArray(validationFiles);

        validBatchSize = config.getInt("validBatchSize", 100);


        visualizationFilename =
                config.getString("visualizationFilename", "src/output/embeddings.csv");

        saveVisualization =
                config.getBoolean("saveVisualization", false);
    }


    private static List<String> getFilesFromDir(String dirName, String suffix) throws IOException {
        List<String> files = new ArrayList<String>();
        if (dirName != null) {
            File folder = new File(dirName);
            File[] listOfFiles;
            if (suffix != null) {
                FileFilter filter = new WildcardFileFilter("*." + suffix);
                listOfFiles = folder.listFiles(filter);
            } else {
                listOfFiles = folder.listFiles();
            }

            for (int i = 0; i < listOfFiles.length; i++) {
                if (listOfFiles[i].isFile()) {
                    files.add(listOfFiles[i].getCanonicalPath());
                }
            }
        }
        return files;
    }
}


