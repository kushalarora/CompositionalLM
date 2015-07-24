package com.kushalarora.compositionalLM.optimizer;

import com.google.common.collect.Lists;
import com.kushalarora.compositionalLM.model.IDerivatives;
import com.kushalarora.compositionalLM.options.Options;
import lombok.extern.slf4j.Slf4j;

import java.util.*;
import java.util.concurrent.*;

/**
 * Created by karora on 7/14/15.
 */
@Slf4j
public abstract class AbstractOptimizer<T extends IIndexed, D extends IDerivatives<T>>
        implements IOptimizer<T, D> {
    private final Random rand;
    protected Options op;
    protected ExecutorService executor;

    D dvAcc;

    protected AbstractOptimizer(Options op, D dvAcc) {
        this.op = op;
        this.dvAcc = dvAcc;
        rand = new Random();
    }

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
                    new ArrayList<Future<Double>>();

            for (final T data : validationSet) {
                log.info("Starting Validation#{}: {}", data.getIndex(), data);
                Future<Double> future = executor.submit(new Callable<Double>() {
                    public Double call() throws Exception {
                        return getValidationScore(data);
                    }
                });
                futureList.add(future);
            }

            int idx = 0;
            Iterator<Future<Double>> it = futureList.iterator();
            while (it.hasNext()) {
                try {
                    Future<Double> future = it.next();
                    Double score = future.get();
                    if (score.isInfinite() || score.isNaN()) {
                        log.info("****Validation#{} is {}****",
                                idx++, score);
                        continue;
                    }

                    log.info("****Finished Validation#{}: {}****",
                            idx++, score);

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
            List<Future<D>> futureList =
                    new ArrayList<Future<D>>();

            for (final T sample : trainBatch) {
                log.info("****Started Training#{}: {}****",
                        sample.getIndex(), sample);
                Future<D> future = executor.submit(
                        new Callable<D>() {
                            public D call() throws Exception {
                                return (D) fitOne(sample);
                            }
                        });
                futureList.add(future);
            }

            Iterator<Future<D>> it = futureList.iterator();
            while (it.hasNext()) {
                try {
                    Future<D> future = it.next();
                    D derivatives = future.get();
                    calcLearningRate(derivatives);
                    derivativeAcc(derivatives);
                    log.info("****Finished Training#{}****",
                            derivatives.getData().getIndex());
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } catch (ExecutionException e) {
                    e.printStackTrace();
                }
                it.remove();
            }
        } else {
            for (T sample : trainBatch) {
                log.info("****Started Training#{}: {}****",
                        sample.getIndex(), sample);

                D derivatives = fitOne(sample);
                calcLearningRate(derivatives);
                derivativeAcc(derivatives);
                log.info("****Finished Training#{}****",
                        sample.getIndex());
            }
        }
        D accDv = getAccumulatedDerivative();
        accDv.mul(1.0/trainBatch.size());
        updateParams(accDv);
        clearLearningRate();
        flushDerivaiveAcc();
    }

    public D fitOne(T data) {
        return calcDerivative(data);
    }

    public void fit(Iterable<T> trainSet, Iterable<T> validationSet) {
        fit(Lists.<T>newArrayList(trainSet),
                Lists.<T>newArrayList(validationSet));

    }

    public synchronized void derivativeAcc(D derivatives) {
        dvAcc.add(derivatives);
    }

    public D getAccumulatedDerivative() {
        return dvAcc;
    }

    public synchronized void flushDerivaiveAcc() {
        dvAcc.clear();
    }

    public void fit(List<T> trainSet) {
        fit(trainSet, new ArrayList<T>());
    }

    public void fit(Iterable<T> trainSet) {
        fit(Lists.newArrayList(trainSet));
    }
}
