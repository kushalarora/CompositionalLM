package com.kushalarora.compositionalLM.optimizer;

import com.google.common.base.Function;
import com.kushalarora.compositionalLM.derivatives.IDerivatives;
import com.kushalarora.compositionalLM.options.Options;
import lombok.extern.slf4j.Slf4j;

import javax.annotation.Nullable;
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


    private Function<List<T>, Void> fitRoutineParallel =
            new Function<List<T>, Void>() {
                @Nullable
                public Void apply(@Nullable List<T> trainBatch) {
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
                    return null;
                }
            };

    Function<List<T>, Void> fitRoutineSeq =
            new Function<List<T>, Void>() {
                @Nullable
                public Void apply(@Nullable List<T> trainBatch) {
                    for (T sample : trainBatch) {
                        log.info("****Started Training#{}: {}****",
                                sample.getIndex(), sample);

                        D derivatives = fitOne(sample);
                        calcLearningRate(derivatives);
                        derivativeAcc(derivatives);
                        log.info("****Finished Training#{}****",
                                sample.getIndex());
                    }
                    return null;
                }
            };

    Function<List<T>, Double> validRoutineParallel =
            new Function<List<T>, Double>() {
                @Nullable
                public Double apply(final @Nullable List<T> validList) {
                    double validationScore = 0;
                    List<Future<Double>> futureList =
                            new ArrayList<Future<Double>>();

                    for (final T data : validList) {
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
                    return validationScore;
                }
            };

    Function<List<T>, Double> validRoutineSeq =
            new Function<List<T>, Double>() {
                @Nullable
                public Double apply(@Nullable List<T> validList) {
                    double validationScore = 0;
                    int count = 0;
                    int idx = 0;
                    for (final T data : validList) {
                        log.info("Validation#{}: {}", idx++, data);
                        Double score = getValidationScore(data);
                        if (score.isInfinite() || score.isNaN()) {
                            log.info("******** Validation#{} is {}************", data.getIndex(), score);
                            continue;
                        }
                        log.info("*********Finished Validation#{}: {} ************", data.getIndex(), score);
                        validationScore += score;
                    }

                    return validationScore;
                }
            };

    public D fitOne(T data) {
        return calcDerivative(data);
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

    public void fit(List<? extends List<T>> trainSet, List<? extends List<T>> validSet) {

        Function<List<T>, Void> trainFunction;
        Function<List<T>, Double> validFunction;
        if (op.trainOp.parallel) {
            log.info("Running in parallel mode");
            log.info("NumThreads#: {}", op.trainOp.nThreads);
            executor =
                    Executors.newFixedThreadPool(op.trainOp.nThreads);
            trainFunction = fitRoutineParallel;
            validFunction = validRoutineParallel;
        } else {
            trainFunction = fitRoutineSeq;
            validFunction = validRoutineSeq;
        }

        int iter = 0;
        int epoch = 0;
        boolean done = false;
        double bestValidationScore = Double.MAX_VALUE;

        // do training these many times
        while (epoch < op.trainOp.maxEpochs && !done) {

            // process all these lists
            for (int trainListIdx = 0; trainListIdx < trainSet.size(); trainListIdx++) {

                List<T> trainList = trainSet.get(trainListIdx);

                int numBatches = trainList.size() / op.trainOp.batchSize + 1;

                // shuffle to avoid overfitting
                List<T> shuffledSet = new ArrayList<T>(trainList);
                Collections.shuffle(shuffledSet, rand);


                for (int batchIdx = 0; batchIdx < numBatches; batchIdx++) {

                    log.info("Starting trainList: {} batch#: {}", trainListIdx, batchIdx);

                    // get batch
                    int startIdx = batchIdx * op.trainOp.batchSize;
                    int endIdx = (batchIdx + 1) * op.trainOp.batchSize;
                    if (endIdx > shuffledSet.size()) {
                        endIdx = shuffledSet.size();
                    }

                    int batchSize = endIdx - startIdx;

                    // In case there batch size is multiple of actual size
                    // we would have a case of blank sentence
                    if (startIdx >= endIdx) {
                        continue;
                    }

                    // train batch
                    trainFunction.apply(shuffledSet.subList(startIdx, endIdx));

                    // normalize accumulated derivative
                    D accDv = getAccumulatedDerivative();
                    accDv.mul(1.0 / batchSize);

                    // update param for this batch
                    updateParams(accDv);

                    // clear accumulator and
                    // re-initialize learing rate
                    clearLearningRate();
                    flushDerivaiveAcc();

                    // shall validate?
                    if (op.trainOp.validate &&
                            (iter + 1) % op.trainOp.validationFreq == 0) {

                        // calc mean for validation set
                        double cumlScore = 0;
                        double cumlSize = 0;
                        for (int validListIdx = 0; validListIdx < validSet.size(); validListIdx++) {
                            List<T> validList = validSet.get(validListIdx);

                            int validNumBatches = validList.size()/op.trainOp.validBatchSize;

                            for (int validBatchIdx = 0; validBatchIdx < validNumBatches; validBatchIdx++) {

                                log.info("Starting validList#: {}, validBatch#: {}", validListIdx, validBatchIdx);

                                // get batch
                                int validStartIdx = validBatchIdx * op.trainOp.validBatchSize;
                                int validEndIdx = (validBatchIdx + 1) * op.trainOp.validBatchSize;
                                if (validEndIdx > validList.size()) {
                                    validEndIdx = validList.size();
                                }

                                int validBatchSize = validEndIdx - validStartIdx;

                                // In case there batch size is multiple of actual size
                                // we would have a case of blank sentence
                                if (validStartIdx >= validEndIdx) {
                                    continue;
                                }

                                cumlScore += validFunction.apply(
                                        validList.subList(validStartIdx, validEndIdx));

                                cumlSize += validBatchSize;
                            }
                        }

                        double mean = cumlScore / cumlSize;
                        log.info("Mean validation score iter#{}: {}", iter, mean);

                        // is better than the bestt
                        if (mean < bestValidationScore) {

                            // save model
                            bestValidationScore = mean;
                            log.info("Updated best validation score");
                            saveModel();

                            // good enough for us?
                            if (mean > bestValidationScore * (1 - op.trainOp.tolerance)) {
                                done = true;
                                log.info("Done training");
                            } // end if mean > bestValidationScore

                        } // end if mean < bestValidationScore
                    }   // end if validate

                    // this iteration done
                    iter += 1;
                } // end for batch
                epoch += 1;
            }   // end for trainList
        }   // end while epoch


        // shutdown if parallel
        if (op.trainOp.parallel) {
            executor.shutdown();
        }

    }
}
