package com.kushalarora.compositionalLM.optimizer;

import com.kushalarora.compositionalLM.lang.Sentence;
import com.kushalarora.compositionalLM.model.IDerivatives;
import com.kushalarora.compositionalLM.options.Options;
import lombok.extern.slf4j.Slf4j;

/**
 * Created by karora on 7/14/15.
 */

@Slf4j
public abstract class AbstractSGDOptimizer<T extends IIndexed, D extends IDerivatives<T>>
        extends AbstractOptimizer<T, D> {
    private int count;
    D dvAcc;

    protected AbstractSGDOptimizer(Options op, D dvAcc) {
        super(op, dvAcc);
        count = 0;
    }

    public void updateParams(D derivatives) {
        derivatives.mul(-1 * op.trainOp.learningRate / count);
        getParams()
                .update(derivatives);
    }

    public void calcLearningRate(final D derivatives) {
        count++;
    }

    public void clearLearningRate() {
        count = 0;
    }
}
