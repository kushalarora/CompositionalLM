package com.kushalarora.compositionalLM.optimizer;

import com.kushalarora.compositionalLM.derivatives.IDerivative;
import com.kushalarora.compositionalLM.model.IParameter;
import com.kushalarora.compositionalLM.model.IParameterDerivatives;
import com.kushalarora.compositionalLM.options.Options;

import java.util.List;

/**
 * Created by karora on 7/14/15.
 */
public abstract class SGDOptimizer<T> extends AbstractOptimizer<T> {


    protected SGDOptimizer(Options op) {
        super(op);
    }

    @Override
    public void fitRoutine(List<T> trainBatch) {
        IParameterDerivatives derivative;
        derivative = calcDerivative(trainBatch.get(0));
        for (int i = 1; i < trainBatch.size(); i++) {
            derivative.add(calcDerivative(trainBatch.get(i)));
        }
        IParameter params = getParams();
        params.update(derivative.mul(-1 * op.trainOp.learningRate));
    }

}
