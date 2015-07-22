package com.kushalarora.compositionalLM.optimizer;

import com.google.common.collect.Lists;
import com.kushalarora.compositionalLM.model.AbstractDerivatives;
import com.kushalarora.compositionalLM.model.IParameter;
import com.kushalarora.compositionalLM.model.IParameterDerivatives;
import com.kushalarora.compositionalLM.options.Options;
import lombok.extern.slf4j.Slf4j;

import java.util.*;
import java.util.concurrent.*;

/**
 * Created by karora on 7/14/15.
 */
@Slf4j
public abstract class AbstractOptimizer<T extends IIndexed> implements IOptimizer<T> {
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

                // In case there batch size is multiple of actual size
                // we would have a case of blank sentence
                if (startIdx >= endIdx) {
                    continue;
                }

                fitRoutine(trainSet.subList(startIdx, endIdx));

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
                }   // end if validate
            }   // end for batch < numBatch
            epoch += 1;
        }   // end  while epoch

        if (op.trainOp.parallel) {
            executor.shutdown();
        }
    }

    private double validationRoutine(List<T> validationSet) {
        double validationScore = 0;
        if (op.trainOp.parallel) {

            List<Future<Double>> futureList =
                    new ArrayList<Future<Double>> ();

            for (final T data : validationSet) {
                log.info("Starting Validation#{}: {}", data.getIndex(), data);
                Callable<Double> callable =
                        new Callable<Double>() {
                            public Double call() throws Exception {
                                return getValidationScore(data);
                            }
                        };
                Future<Double> future = executor.submit(callable);
                futureList.add(future);
            }

            int idx = 0;
            Iterator<Future<Double>> it = futureList.iterator();
            while(it.hasNext()) {
                try {
                    Future<Double> future = it.next();
                    Double score = future.get();
                    if (score.isInfinite() || score.isNaN()) {
                        log.info("******** Validation#{} is {}************", idx++, score);
                        continue;
                    }
                    log.info("*********Finished Validation#{}: {} ************", idx++, score);
                    validationScore += score;
                    it.remove();
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
                Double score = getValidationScore(data);
                if (score.isInfinite() || score.isNaN()) {
                    log.info("******** Validation#{} is {}************", data.getIndex(), score);
                    continue;
                }
                log.info("*********Finished Validation#{}: {} ************", data.getIndex(), score);
                validationScore += score;
            }
        }
        return validationScore / validationSet.size();
    }

    public void fitRoutine(List<T> trainBatch) {
        if (op.trainOp.parallel) {
            List<Future<AbstractDerivatives<T>>> futureList =
                    new ArrayList<Future<AbstractDerivatives<T>>>();

            for (final T sample : trainBatch) {
                log.info("*********Started Training#{}: {} ************", sample.getIndex(), sample);
                Callable<AbstractDerivatives<T>> callable =
                        new Callable<AbstractDerivatives<T>>() {
                            public AbstractDerivatives<T> call() throws Exception {
                                return (AbstractDerivatives<T>)fitOne(sample);
                            }
                        };
                Future<AbstractDerivatives<T>> future =
                        executor.submit(callable);
                futureList.add(future);
            }

            Iterator<Future<AbstractDerivatives<T>>> it = futureList.iterator();
            while(it.hasNext())  {
                try {
                    Future<AbstractDerivatives<T>> future = it.next();
                    AbstractDerivatives<T> derivatives =
                            future.get();
                    calcLearningRate(derivatives);
                    derivativeAccumulator(derivatives);
                    log.info("*********Finished Training#{} ************", derivatives.getData().getIndex());
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } catch (ExecutionException e) {
                    e.printStackTrace();
                }
                it.remove();
            }
        } else {
            for (T sample : trainBatch) {
                log.info("*********Started Training#{}: {} ************", sample.getIndex(), sample);
                AbstractDerivatives derivatives = (AbstractDerivatives<T>)fitOne(sample);
                calcLearningRate(derivatives);
                derivativeAccumulator(derivatives);
                log.info("*********Finished Training#{} ************", sample.getIndex());
            }
        }

        updateParams(getAccumulatedDerivative());
        clearLearningRate();
        flushDerivaiveAccumulator();
    }

    public IParameterDerivatives<T> fitOne(T data) {
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
