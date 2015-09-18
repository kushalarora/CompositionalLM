package com.kushalarora.compositionalLM.optimizer;

import com.kushalarora.compositionalLM.derivatives.IDerivatives;
import com.kushalarora.compositionalLM.options.Options;
import org.nd4j.linalg.ops.transforms.Pow;

import static org.nd4j.linalg.ops.transforms.Transforms.pow;
import static org.nd4j.linalg.ops.transforms.Transforms.sqrt;

/**
 * Created by karora on 7/22/15.
 */
public abstract class AbstractAdaGradOptimzer<T extends IIndexed, D extends IDerivatives<T>>
        extends AbstractOptimizer<T, D> {

    D dvGrad;

    protected AbstractAdaGradOptimzer(Options op, D dvAcc, D dvGrad) {
        super(op, dvAcc);
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
