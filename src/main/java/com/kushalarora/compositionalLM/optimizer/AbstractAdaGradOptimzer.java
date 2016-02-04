package com.kushalarora.compositionalLM.optimizer;

import com.kushalarora.compositionalLM.derivatives.IDerivatives;
import com.kushalarora.compositionalLM.documentprocessor.DocumentProcessorWrapper;
import com.kushalarora.compositionalLM.options.Options;
import com.kushalarora.compositionalLM.utils.Parallelizer;

/**
 * Created by karora on 7/22/15.
 */
public abstract class AbstractAdaGradOptimzer<T extends IIndexedSized, D extends IDerivatives>
        extends AbstractOptimizer<T, D> {

    D dvGrad;

    protected AbstractAdaGradOptimzer(Options op, D dvAcc, D dvGrad, Parallelizer parallelizer) {
        super(op, dvAcc, parallelizer);
        this.dvGrad = dvGrad;
    }

    public void updateParams(D derivatives) {
        derivatives = (D) dvGrad.adaGrad(derivatives);
        derivatives.mul(-1 * op.trainOp.learningRate);
        getParams().update(derivatives);
    }

    public void clearLearningRate() {
        // do nothing
    }

    public void calcLearningRate(D derivatives) {
    }
}
