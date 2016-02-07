package com.kushalarora.compositionalLM.optimizer;

import com.kushalarora.compositionalLM.derivatives.IDerivatives;
import com.kushalarora.compositionalLM.documentprocessor.DocumentProcessorWrapper;
import com.kushalarora.compositionalLM.options.Options;
import com.kushalarora.compositionalLM.utils.Parallelizer;
import lombok.extern.slf4j.Slf4j;

/**
 * Created by karora on 7/14/15.
 */

@Slf4j
public abstract class AbstractSGDOptimizer<T extends IIndexedSized, D extends IDerivatives>
        extends AbstractOptimizer<T, D> {
    private int count;
    D dvAcc;

    protected AbstractSGDOptimizer(Options op, D dvAcc, Parallelizer parallelizer) {
        super(op, dvAcc, parallelizer);
        count = 0;
    }

    public void updateParams(D derivatives) {
        derivatives.mul(-1 * op.trainOp.learningRate);
        getParams().update(derivatives);
    }

    public void clearLearningRate() {

    }
}
