package com.kushalarora.compositionalLM.optimizer;

import com.kushalarora.compositionalLM.options.Options;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Created by karora on 7/14/15.
 */
public abstract class AbstractOptimizer<T> implements IOptimizer {
    private final Random rand;
    protected Options op;

    protected AbstractOptimizer(Options op) {
        this.op = op;
        rand = new Random();
    }

    public abstract double getValidationScore(T data);

    public abstract void saveModel();

    public void fit(List<T> trainSet, List<T> validationSet) {
        boolean done = false;
        int numBatch = trainSet.size() / op.trainOp.batchSize + 1;
        int epoch = 0;
        double validationScore = 0;
        double bestValidationScore = 0;
        while (epoch < op.trainOp.maxEpochs && !done) {
            List<T> shuffledSet = new ArrayList<T>(trainSet);
            Collections.shuffle(shuffledSet, rand);

            for (int batch = 0; batch < numBatch; batch++) {

                int iter = epoch * op.trainOp.batchSize + batch;

                int startIdx = batch * op.trainOp.batchSize;
                int endIdx = (batch + 1) * op.trainOp.batchSize;
                if (endIdx > trainSet.size()) {
                    endIdx = trainSet.size();
                }

                fitRoutine(trainSet.subList(startIdx, endIdx));


                if (op.trainOp.validate &&
                        (iter + 1) % op.trainOp.validationFreq == 0) {

                    validationScore = 0;
                    for (T data : validationSet) {
                        validationScore += getValidationScore(data);
                    }
                    double mean = validationScore / validationSet.size();

                    if (mean < bestValidationScore) {
                        if (mean < bestValidationScore * (1 - op.trainOp.tolerance)) {
                            done = true;
                        }
                        bestValidationScore = mean;
                        saveModel();
                    }
                }
            }
            epoch += 1;
        }
    }

    public abstract void fitRoutine(List<T> trainBatch);

}
