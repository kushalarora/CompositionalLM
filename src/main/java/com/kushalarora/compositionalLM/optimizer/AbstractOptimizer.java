package com.kushalarora.compositionalLM.optimizer;

import com.google.common.collect.Lists;
import com.kushalarora.compositionalLM.model.IParameterDerivatives;
import com.kushalarora.compositionalLM.options.Options;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.*;

/**
 * Created by karora on 7/14/15.
 */
@Slf4j
public abstract class AbstractOptimizer<T> implements IOptimizer<T> {
    private final Random rand;
    protected Options op;
    protected ExecutorService executor;

    protected AbstractOptimizer(Options op) {
        this.op = op;
        rand = new Random();
    }

    public abstract double getValidationScore(T data);

    public abstract void saveModel();

    public void fit(List<T> trainSet, List<T> validationSet) {
        if (op.trainOp.parallel) {
            log.info("Running in parallel mode");
            log.info("NumThreads#: {}", op.trainOp.nThreads);
            executor =
                    Executors.newFixedThreadPool(op.trainOp.nThreads);
        }

        boolean done = false;
        int numBatch = trainSet.size() / op.trainOp.batchSize + 1;
        int epoch = 0;
        double bestValidationScore = Double.MAX_VALUE;
        while (epoch < op.trainOp.maxEpochs) {
            List<T> shuffledSet = new ArrayList<T>(trainSet);
            Collections.shuffle(shuffledSet, rand);

            for (int batch = 0; batch < numBatch; batch++) {

                int iter = epoch * op.trainOp.batchSize + batch;

                int startIdx = batch * op.trainOp.batchSize;
                int endIdx = (batch + 1) * op.trainOp.batchSize;
                if (endIdx > trainSet.size()) {
                    endIdx = trainSet.size();
                }

                fitRoutine(startIdx, trainSet.subList(startIdx, endIdx));


                if (op.trainOp.validate &&
                        (iter + 1) % op.trainOp.validationFreq == 0) {
                    double mean = validationRoutine(validationSet);
                    log.info("Mean validation score iter#{}: {}", iter, mean);

                    if (mean < bestValidationScore) {
                        // TODO Fix this
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

    private double validationRoutine(List<T> validationSet) {
        double validationScore = 0;
        if (op.trainOp.parallel) {

            List<Future<Double>> futureList =
                    new ArrayList<Future<Double>> ();

            int idx = 0;
            for (final T data : validationSet) {
                log.info("Starting Validation#{}: {}", idx++, data);
                Callable<Double> callable =
                        new Callable<Double>() {
                            public Double call() throws Exception {
                                return getValidationScore(data);
                            }
                        };
                Future<Double> future = executor.submit(callable);
                futureList.add(future);
            }

            idx = 0;
            for (Future<Double> future : futureList) {
                try {
                    Double score = future.get();
                    if (score.isInfinite() || score.isNaN()) {
                        log.info("******** Validation#{} is {}************", idx++, score);
                        continue;
                    }
                    log.info("*********Finished Validation#{}: {} ************", idx++, score);
                    validationScore += future.get();

                } catch (InterruptedException e) {
                    e.printStackTrace();
                } catch (ExecutionException e) {
                    e.printStackTrace();
                }

            }

        } else {

            int idx = 0;
            for (T data : validationSet) {
                log.info("Validation#{}: {}", idx++, data);
                validationScore += getValidationScore(data);
            }
        }
        return validationScore / validationSet.size();
    }

    public void fitRoutine(int startIdx, List<T> trainBatch) {
        int idx = startIdx;
        IParameterDerivatives derivative = null;

        if (op.trainOp.parallel) {
            List<Future<IParameterDerivatives<T>>> futureList =
                    new ArrayList<Future<IParameterDerivatives<T>>>();

            for (final T sample : trainBatch) {
                log.info("*********Started Training#{}: {} ************", idx++, sample);
                Callable<IParameterDerivatives<T>> callable =
                        new Callable<IParameterDerivatives<T>>() {
                            public IParameterDerivatives<T> call() throws Exception {
                                return fitOne(sample);
                            }
                        };
                Future<IParameterDerivatives<T>> future =
                        executor.submit(callable);
                futureList.add(future);
            }

            idx = startIdx;
            for (Future<IParameterDerivatives<T>> future : futureList) {
                try {
                    IParameterDerivatives<T> derivatives =
                            future.get();
                    T sample = derivatives.getSentence();
                    calcLearningRate(sample, derivatives);
                    derivativeAccumulator(derivatives);

                    log.info("*********Finished Training#{} ************", idx++);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } catch (ExecutionException e) {
                    e.printStackTrace();
                }
            }
        } else {
            for (T sample : trainBatch) {
                IParameterDerivatives derivatives = fitOne(sample);
                calcLearningRate(sample, derivatives);
                derivativeAccumulator(derivatives);
            }
        }

        updateParams(getAccumulatedDerivative());
        flushDerivaiveAccumulator();
    }

    public IParameterDerivatives fitOne(T data) {

        return calcDerivative(data);
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
