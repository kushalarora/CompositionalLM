package com.kushalarora.compositionalLM.optimizer;

import com.kushalarora.compositionalLM.derivatives.IDerivatives;
import com.kushalarora.compositionalLM.documentprocessor.DocumentProcessorWrapper;
import com.kushalarora.compositionalLM.options.Options;

/**
 * Created by karora on 7/22/15.
 */
public abstract class AbstractAdaGradOptimzer<T extends IIndexedSized, D extends IDerivatives<T>>
        extends AbstractOptimizer<T, D> {

    D dvGrad;

    protected AbstractAdaGradOptimzer(Options op, D dvAcc, D dvGrad, DocumentProcessorWrapper<T> documentProcessorWrapper) {
        super(op, dvAcc, documentProcessorWrapper);
        this.dvGrad = dvGrad;
    }

    public void updateParams(D derivatives) {
        derivatives = (D) dvGrad.adaGrad(derivatives);
        derivatives.mul(op.trainOp.learningRate);
        getParams().update(derivatives);
    }

    public void clearLearningRate() {
        // do nothing
    }

    public void calcLearningRate(D derivatives) {
    }
}
