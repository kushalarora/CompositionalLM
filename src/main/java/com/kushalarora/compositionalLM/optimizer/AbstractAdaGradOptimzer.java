package com.kushalarora.compositionalLM.optimizer;

import com.kushalarora.compositionalLM.model.IDerivatives;
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
        this.dvGrad.add(1);
    }

    public void updateParams(D derivatives) {
        D newD = (D)derivatives.deepcopy();
        newD.power(2);
        dvGrad.add(newD);

        D dvGradCopy = (D) dvGrad.deepcopy();
        dvGradCopy.power(-0.5);
        dvGradCopy.add(1e-5);
        derivatives.mul(op.trainOp.learningRate);
        derivatives.mul(dvGradCopy);
        getParams().update(derivatives);
    }

    public void clearLearningRate() {
        // do nothing
    }

    public void calcLearningRate(D derivatives) {
    }
}
