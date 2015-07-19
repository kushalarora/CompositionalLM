package com.kushalarora.compositionalLM.optimizer;

import com.google.common.collect.Lists;
import com.kushalarora.compositionalLM.options.Options;
import lombok.extern.slf4j.Slf4j;

import java.util.*;

/**
 * Created by karora on 7/14/15.
 */
@Slf4j
public abstract class AbstractOptimizer<T> implements IOptimizer<T> {
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
        double bestValidationScore = Double.MAX_VALUE;
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


                if (true || op.trainOp.validate &&
                        (iter + 1) % op.trainOp.validationFreq == 0) {

                    validationScore = 0;
                    for (T data : validationSet) {
                        log.info("Processing Validation set for {}", data);
                        validationScore += getValidationScore(data);
                    }
                    double mean = validationScore / validationSet.size();
                    log.info("new validation score for iter : {} ==> {}", iter, mean);

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

    public void fit(Iterable<T> trainSet, Iterable<T> validationSet) {
        fit(Lists.<T>newArrayList(trainSet),
                Lists.<T>newArrayList(validationSet));

    }

    public void fit(List<T> trainSet) {
        fit(trainSet, new ArrayList<T>());
    }

    public void fit(Iterable<T> trainSet) {
        fit(Lists.newArrayList(trainSet));
    }
}