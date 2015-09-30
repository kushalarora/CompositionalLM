package com.kushalarora.compositionalLM.optimizer;

import com.google.common.base.Function;
import com.kushalarora.compositionalLM.derivatives.IDerivatives;
import com.kushalarora.compositionalLM.documentprocessor.DocumentProcessorWrapper;
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
    protected int iter;
    protected int epoch;
    protected double bestValidationScore;
    protected boolean done;
    private DocumentProcessorWrapper<T> documentProcessor;

    D dvAcc;

    protected AbstractOptimizer(Options op, D dvAcc, DocumentProcessorWrapper<T> documentProcessor) {
        this.op = op;
        this.dvAcc = dvAcc;
        this.documentProcessor = documentProcessor;
        rand = new Random();
        iter = 0;
        epoch = 0;
        bestValidationScore = Double.MAX_VALUE;
        done = false;

    }


    private Function<List<T>, Double> fitRoutineParallel =
            new Function<List<T>, Double>() {
                @Nullable
                public Double apply(@Nullable List<T> trainBatch) {
                    List<Future<D>> futureList =
                            new ArrayList<Future<D>>();
                    double cumlTrainingScore = 0.0;
                    int cumlTrainingSize = 0;
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
                            cumlTrainingScore += derivatives.getScore();
                            cumlTrainingSize += 1;
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
                    return cumlTrainingScore/cumlTrainingSize;
                }
            };

    Function<List<T>, Double> fitRoutineSeq =
            new Function<List<T>, Double>() {
                @Nullable
                public Double apply(@Nullable List<T> trainBatch) {
                    double cumlTrainingScore = 0.0;
                    int cumlTrainingSize = 0;
                    for (T sample : trainBatch) {
                        log.info("****Started Training#{}: {}****",
                                sample.getIndex(), sample);

                        D derivatives = fitOne(sample);
                        cumlTrainingScore += derivatives.getScore();
                        cumlTrainingSize += 1;
                        calcLearningRate(derivatives);
                        derivativeAcc(derivatives);
                        log.info("****Finished Training#{}****",
                                sample.getIndex());
                    }
                    return cumlTrainingScore/cumlTrainingSize;
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

    private double getValidationScore(Function<List<T>, Double> validFunction, List<String> validFileList) {
        // calc mean for validation set
        double cumlScore = 0;
        double cumlSize = 0;
        for (int validListIdx = 0; validListIdx < validFileList.size(); validListIdx++) {
            String validFilename = validFileList.get(validListIdx);

            Iterator<T> validIter = documentProcessor.getIterator(validFilename);
            List<T> validList = new ArrayList<T>();

            int validBatchIdx = 0;
            while (validIter.hasNext()) {
                validList.clear();
                log.info("Starting validList#: {},  validBatch#: {}", validListIdx, validBatchIdx);

                for (int idx = 0; idx < op.trainOp.validBatchSize && validIter.hasNext(); idx++) {
                    validList.add(validIter.next());
                }

                int validBatchSize = validList.size();

                cumlScore += validFunction.apply(validList);

                cumlSize += validBatchSize;

                validBatchIdx++;
            }

        }

        return cumlScore / cumlSize;
    }

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

    public void fit(List<String> trainFileList, List<String> validSet) {

        Function<List<T>, Double> trainFunction;
        Function<List<T>, Double> validFunction;
        if (op.trainOp.parallel) {
            log.info("Running in parallel mode");
            log.info("NumThreads#: {}", op.trainOp.nThreads);
            executor =
                    Executors.newFixedThreadPool(2 * op.trainOp.nThreads);
            trainFunction = fitRoutineParallel;
            validFunction = validRoutineParallel;
        } else {
            trainFunction = fitRoutineSeq;
            validFunction = validRoutineSeq;
        }

        log.info("Intial validation score#: {}",
                getValidationScore(validFunction, validSet));

        epoch = 0;
        iter = 0;
        bestValidationScore = Double.MAX_VALUE;
        done = false;

        // do training these many times
        while (epoch < op.trainOp.maxEpochs && !done) {
            double cumlTrainScore = 0.0;
            int cumlTrainBatch = 0;

            log.info("Starting epoch: {}", epoch);
            List<T> trainList = new ArrayList<T>();


            // process all these lists
            for (int trainFileIdx = 0; trainFileIdx < trainFileList.size(); trainFileIdx++) {

                Iterator<T> trainIter = null;
                String trainFilename = trainFileList.get(trainFileIdx);
                trainIter = documentProcessor.getIterator(trainFilename);

                int batchIdx = 0;

                while (trainIter.hasNext()) {
                    trainList.clear();


                    for (int idx = 0; idx < op.trainOp.batchSize && trainIter.hasNext(); idx++) {
                        trainList.add(trainIter.next());
                    }

                    int batchSize = trainList.size();
                    Collections.shuffle(trainList, rand);

                    log.info("Starting epoch#: {}, trainList: {} , batch#: {}", epoch, trainFileIdx, batchIdx);

                    // train batch
                    double score = trainFunction.apply(trainList);

                    cumlTrainScore += score * batchSize;
                    cumlTrainBatch += batchSize;

                    log.info("Training score epoch#: {}, trainList: {} , batch#: {} => {}",
                            epoch, trainFileIdx, batchIdx, score);

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

                        double mean = getValidationScore(validFunction, validSet);
                        log.info("Mean validation score epoch#{}, iter#{}: {}",
                                epoch, iter, mean);

                        // is better than the bestt
                        if (mean < bestValidationScore) {

                            // save model
                            bestValidationScore = mean;
                            log.info("Updated best validation score epoch# {}, iter# {}:: {}",
                                    epoch, iter, mean);
                            saveModel(iter, epoch);

                            // good enough for us?
                            if (mean > bestValidationScore * (1 - op.trainOp.tolerance)) {
                                done = true;
                                log.info("Done training");
                            } // end if mean > bestValidationScore

                        } // end if mean < bestValidationScore
                    }   // end if validate

                    // this iteration done
                    iter += 1;
                    batchIdx += 1;
                } // end for batch

                log.info("Training score epoch#: {}, trainList: {}  => {}",
                        epoch, trainFileIdx, cumlTrainScore / cumlTrainBatch);

                epoch += 1;
            }   // end for trainList
        }   // end while epoch


        // shutdown if parallel
        if (op.trainOp.parallel) {
            executor.shutdown();
        }
    }
}
