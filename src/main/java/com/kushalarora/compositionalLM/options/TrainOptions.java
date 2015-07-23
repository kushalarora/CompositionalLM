package com.kushalarora.compositionalLM.options;

import com.kushalarora.compositionalLM.caching.CacheFactory;
import com.kushalarora.compositionalLM.optimizer.OptimizerFactory;
import lombok.ToString;
import org.apache.commons.configuration.Configuration;

import java.io.Serializable;

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

    public TrainOptions(Configuration config) {
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
    }
}


