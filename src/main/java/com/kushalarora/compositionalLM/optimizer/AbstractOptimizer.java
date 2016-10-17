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
    protected boolean done;
    private Parallelizer parallelizer;
    protected D dvAcc;

    protected AbstractOptimizer(Options op, D dvAcc, Parallelizer parallelizer) {
        this.op = op;
        this.dvAcc = dvAcc;
        rand = new Random();
        epoch = 0;
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


    private double fitBatch(final List<T> trainList)
            throws ExecutionException, InterruptedException {
        double score = 0;
        Collections.shuffle(trainList, rand);
        int batchSize = trainList.size();
        Function<Integer, D> fitRoutine =
                new Function<Integer, D>() {
                    @Nullable
                    public D apply(@Nullable Integer integer) {
                        final T data = trainList.get(integer);
                        return fitOne(data);
                    }
                };

        if (op.trainOp.dataParallel) {
            List<Future<List<D>>> futures =
                    parallelizer.parallelizer(0, batchSize, fitRoutine);

            for (Future<List<D>> future : futures) {
                for (D derivative : future.get()) {
                    score += derivative.getScore();
                    derivativeAcc(derivative);
                    derivative.clean();
                    derivative = null;
                }
            }
        } else {
            for (int i = 0; i < batchSize; i++) {
                D derivative = fitRoutine.apply(i);
                derivativeAcc(derivative);
                derivative.clean();
                derivative = null;
            }
        }

        return score;
    }

    public double getTrainBatchScore(final List<T> trainList)
            throws ExecutionException, InterruptedException {
        int batchSize = trainList.size();
        double cumlScore = 0;
        Function<Integer, Double> scoreRountine =
                new Function<Integer, Double>() {
                    @Nullable
                    public Double apply(@Nullable Integer integer) {
                        return getTrainScore(trainList.get(integer));
                    }
                };

        if (op.trainOp.dataParallel) {
            List<Future<List<Double>>> futures =
                    parallelizer.parallelizer(0, batchSize, scoreRountine);

            for (Future<List<Double>> future : futures) {
                for (Double score : future.get()) {
                    cumlScore +=score;
                }
            }
        } else {
            for (int i = 0; i < batchSize; i++) {
                Double score = scoreRountine.apply(i);
                log.info("Training sentence#{}:: {}", trainList.get(i).getIndex(), score);
                cumlScore +=score;
            }
        }

        return cumlScore;
    }

    public void fit(final List<T> trainFileList)
            throws ExecutionException, InterruptedException {
        epoch = 0;
        double prevBatchScore = Double.POSITIVE_INFINITY;
        double learningRate = op.trainOp.learningRate;
        double tolerance = .999;
        done = false;

        // do training these many times
        while (epoch < op.trainOp.maxOptimizerEpochs && !done) {
            long epochStartTime = System.currentTimeMillis();

            int trainBatchSize = 0;
            for (T trainData : trainFileList) {
                trainBatchSize += trainData.getSize();
            }
            log.info("Starting epoch#: {}", epoch);

            preProcessOnBatch();

            // train batch
            fitBatch(trainFileList);

            System.gc();

            double trainBatchScore = getTrainBatchScore(trainFileList)/ trainBatchSize;

            System.gc();

            long estimatedTime = System.currentTimeMillis() - epochStartTime;

            log.info("$Training$ Ending epoch#: {}, time: {} => {}",
                epoch, estimatedTime, trainBatchScore );


            if (trainBatchScore > prevBatchScore * tolerance) {
                log.info("Reached minimum. TrainingBatchScore {} < prevBatchScore {}", trainBatchScore, prevBatchScore);
                done = true;
            }
            prevBatchScore = trainBatchScore;


            // update param for this batch
            D dAcc = getAccumulatedDerivative();
            dAcc.mul(learningRate / trainBatchSize);
            updateParams(dAcc);
            postProcessOnBatch();

            epoch += 1;

            flushDerivaiveAcc();
        }   // end while

        // clear accumulator and
        // re-initialize learing rate
        clearLearningRate();
    }
}
