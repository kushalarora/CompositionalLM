package com.kushalarora.compositionalLM.optimizer;

import com.google.common.base.Function;
import com.google.common.util.concurrent.AtomicDouble;
import com.kushalarora.compositionalLM.derivatives.IDerivatives;
import com.kushalarora.compositionalLM.options.Options;
import com.kushalarora.compositionalLM.utils.Parallelizer;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;

import javax.annotation.Nullable;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

@Slf4j
public abstract class AbstractOptimizer<T extends IIndexedSized, D extends IDerivatives>
        implements IOptimizer<T, D> {
    private final Random rand;
    protected Options op;
    protected ExecutorService executor;
    protected int iter;
    protected int epoch;

    @Getter
    protected double bestValidationScore;
    protected boolean done;
    private Parallelizer parallelizer;

    protected D dvAcc;

    protected AbstractOptimizer(Options op, D dvAcc, Parallelizer parallelizer) {
        this.op = op;
        this.dvAcc = dvAcc;
        rand = new Random();
        iter = 0;
        epoch = 0;
        bestValidationScore = Double.NEGATIVE_INFINITY;
        done = false;
        this.parallelizer = parallelizer;

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

    private double getValidationScore(final List<T> validList) throws ExecutionException, InterruptedException {
        int validBatchSize = validList.size();
        double validBatchScore = 0;
        Function<Integer, Double> validFunc = new Function<Integer, Double>() {
            @Nullable
            public Double apply(@Nullable Integer integer) {
                return getScore(validList.get(integer));
            }
        };
        if (op.trainOp.parallel) {
            List<Future<List<Double>>> validScoreFutures =
                    parallelizer.parallelizer(0, validBatchSize, validFunc);

            for (Future<List<Double>> future : validScoreFutures) {
                List<Double> scoreList = future.get();
                for (double score : scoreList) {
                    validBatchScore += score;
                }
            }
        } else {
            for (int i = 0; i < validBatchSize; i++) {
                validBatchScore += validFunc.apply(i);
            }
        }
        return validBatchScore;
    }

    private void fitBatch(final List<T> trainList) throws ExecutionException, InterruptedException {
        Collections.shuffle(trainList, rand);
        int batchSize = trainList.size();

        Function<Integer, D> fitRoutine =
                new Function<Integer, D>() {
                    @Nullable
                    public D apply(@Nullable Integer integer) {
                        return fitOne(trainList.get(integer));
                    }
                };

        if (op.trainOp.parallel) {
            List<Future<List<D>>> futures =
                    parallelizer.parallelizer(0, batchSize, fitRoutine);

            for (Future<List<D>> future : futures) {
                for (D derivative : future.get()) {
                    derivativeAcc(derivative);
                }
            }
        } else {
            for (int i = 0; i < batchSize; i++) {
                D derivative = fitRoutine.apply(i);
                derivativeAcc(derivative);
            }
        }

        // update param for this batch
        D dAcc = getAccumulatedDerivative();
        dAcc.mul(1.0 / batchSize);
        updateParams(dAcc);
    }

    public void fit(final List<List<T>> trainFileList, final List<List<T>> validSet)
            throws ExecutionException, InterruptedException {
        epoch = 0;
        iter = 0;
        done = false;

        // do training these many times
        while (epoch < op.trainOp.maxOptimizerEpochs && !done) {
            log.info("Starting epoch#: {}", epoch);
            long epochStartTime = System.currentTimeMillis();

            // process all these lists
            for (int trainFileIdx = 0; trainFileIdx < trainFileList.size(); trainFileIdx++) {
                log.info("Starting epoch#: {}, trainList: {}", epoch, trainFileIdx);
                long epochTrainfileTime = System.currentTimeMillis();

                Iterator<T> trainIter = trainFileList.get(trainFileIdx).iterator();

                int batchIdx = 0;
                while (trainIter.hasNext()) {
                    long startTime = System.currentTimeMillis();
                    log.info("Starting epoch#: {}, trainList: {} , batch#: {}",
                            epoch, trainFileIdx, batchIdx);

                    final List<T> trainList = new ArrayList<T>();
                    for (int idx = 0; idx < op.trainOp.batchSize && trainIter.hasNext(); idx++) {
                        T data = trainIter.next();
                        trainList.add(data);
                        log.info("Training sentence#{}:: {}", data.getIndex(), data);

                    }
                    int batchSize = trainList.size();

                    // train batch
                    fitBatch(trainList);

                    long estimatedTime = System.currentTimeMillis() - startTime;
                    log.info("Ending epoch#: {}, trainList: {} , batch#: {} => {}",
                             epoch, trainFileIdx, batchIdx, estimatedTime);

                    // this iteration done
                    iter += 1;
                    batchIdx += 1;

                    // shall validate?
                    if (op.trainOp.validate &&
                            (iter + 1) % op.trainOp.validationFreq == 0) {
                        long validStartTime = System.currentTimeMillis();
                        double cumlTrainScore = 0.0;
                        int cumlTrainBatchSize = 0;
                        for (int fileIdx = 0; fileIdx < trainFileList.size(); fileIdx++) {
                            final int idx = fileIdx;
                            final AtomicDouble atomicBatchScore = new AtomicDouble(0);

                            Function<Integer, Void> scoreFunc = new Function<Integer, Void>() {
                                @Nullable
                                public Void apply(@Nullable Integer index) {
                                    Double score = getScore(trainFileList
                                                                    .get(idx)
                                                                    .get(index));
                                    synchronized (atomicBatchScore) {
                                        atomicBatchScore.addAndGet(score);
                                    }
                                    return null;
                                }
                            };

                            if (op.trainOp.parallel) {
                                parallelizer.parallelizer(0, batchSize, scoreFunc);
                            }
                            else {
                                for (int i = 0; i < batchSize; i++) {
                                    scoreFunc.apply(i);
                                }
                            }

                            double batchScore = atomicBatchScore.get();

                            log.info("Training score epoch#: {}, trainList: {} => {}",
                                     epoch, trainFileIdx, batchScore);

                            cumlTrainBatchSize += batchSize;
                            cumlTrainScore += batchScore;
                        }

                        log.info("$Training$: Training file done epoch#: {}, trainList: {} => {}",
                                    epoch, trainFileIdx, cumlTrainScore / cumlTrainBatchSize);

                        log.info("Starting validation epoch#: {}, iter#: {}",
                                epoch, iter);

                        // calc mean for validation set
                        double cumlValidScore = 0;
                        double cumlValidSize = 0;
                        for (int validListIdx = 0; validListIdx < validSet.size(); validListIdx++) {
                            Iterator<T> validIter = validSet.get(validListIdx).iterator();
                            int validBatchIdx = 0;
                            while (validIter.hasNext()) {
                                final List<T> validList = new ArrayList<T>();
                                log.info("Starting validList#: {},  validBatch#: {}", validListIdx, validBatchIdx);

                                for (int idx = 0; idx < op.trainOp.validBatchSize && validIter.hasNext(); idx++) {
                                    validList.add(validIter.next());
                                }
                                int validBatchSize = validList.size();

                                cumlValidScore += getValidationScore(validList);

                                cumlValidSize += validBatchSize;
                                validBatchIdx++;
                            }
                        }
                        double mean = cumlValidScore / cumlValidSize;

                        long estimatedValidTime = System.currentTimeMillis() - validStartTime;
                        log.info("$Validation$ Mean validation score epoch#{}, iter#{}, time#{}: {}",
                                epoch, iter, estimatedValidTime, mean);

                        // is better than the bestt
                        if (mean > bestValidationScore) {

                            // good enough for us?
                            if (mean < bestValidationScore * (1 + op.trainOp.tolerance)) {
                                done = true;
                                log.info("Done training mean : {} bestScore: {}", mean, bestValidationScore);
                            } // end if mean > bestValidationScore

                            // save model
                            bestValidationScore = mean;
                            log.info("$Updated Validation$  Updated best validation score epoch# {}, iter# {}:: {}",
                                    epoch, iter, mean);
                            saveModel(iter, epoch);


                        } // end if mean < bestValidationScore
                    }   // end if validate
                } // end for batch

                double estimatedTrainfileTime = System.currentTimeMillis() - epochTrainfileTime;
                log.info("Ending epoch#: {}, trainList: {} => {}",
                         epoch, trainFileIdx, estimatedTrainfileTime);
            }   // end for trainList

            // clear accumulator and
            // re-initialize learing rate
            clearLearningRate();
            flushDerivaiveAcc();

            long estimatedEpochTime = System.currentTimeMillis() - epochStartTime;
            log.info("Training score epoch#: {},  time: {}", epoch, estimatedEpochTime);
            epoch += 1;
        }   // end while epoch
    }
}
