package com.kushalarora.compositionalLM.optimizer;

import com.kushalarora.compositionalLM.derivatives.IDerivatives;
import com.kushalarora.compositionalLM.options.Options;
import com.kushalarora.compositionalLM.utils.Parallelizer;

public abstract class AbstractAdaGradOptimzer<T extends IIndexedSized, D extends IDerivatives>
        extends AbstractOptimizer<T, D> {

    D dvGrad;

    protected AbstractAdaGradOptimzer(Options op, D dvAcc, D dvGrad, Parallelizer parallelizer) {
        super(op, dvAcc, parallelizer);
        this.dvGrad = dvGrad;
    }

    public void updateParams(D derivatives) {
        getParams().update(dvGrad.adaGrad(derivatives));
    }

    public void clearLearningRate() {
        dvGrad.clear();
    }
}
