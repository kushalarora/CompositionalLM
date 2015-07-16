package com.kushalarora.compositionalLM.options;

import lombok.ToString;

import java.io.Serializable;

/**
 * Created by karora on 6/14/15.
 */
@ToString
public class TrainOptions implements Serializable {
    public String[] trainFiles = null;
    public boolean train = false;

    public boolean validate = false;
    public String[] validationFiles = null;
    public int maxEpochs = 50;
    public double tolerance = 1e-3;
    public int batchSize = 50;
    public int validationFreq = 1000;
    public double learningRate = 0.01;
}


