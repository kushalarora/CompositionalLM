package com.kushalarora.compositionalLM.optimizer;

import com.kushalarora.compositionalLM.derivatives.IDerivative;
import com.kushalarora.compositionalLM.model.IParameter;
import com.kushalarora.compositionalLM.model.IParameterDerivatives;
import com.kushalarora.compositionalLM.options.Options;
import lombok.extern.slf4j.Slf4j;

import java.util.Iterator;
import java.util.List;

/**
 * Created by karora on 7/14/15.
 */

@Slf4j
public abstract class AbstractSGDOptimizer<T> extends AbstractOptimizer<T> {


    protected AbstractSGDOptimizer(Options op) {
        super(op);
    }

    public void fitRoutine(List<T> trainBatch) {
        IParameterDerivatives derivative = null;
        for (T sample : trainBatch) {
            derivative = fitOne(sample, derivative);
        }
        IParameter params = getParams();
        params.update(
                derivative.mul(
                        -1 * op.trainOp.learningRate));
    }


    public IParameterDerivatives fitOne(T data, IParameterDerivatives oldDerivative) {
        log.info("*********Training: {} ************", data);
        IParameterDerivatives derivative = calcDerivative(data);
        if (oldDerivative == null) {
            oldDerivative = derivative;
        } else {
            oldDerivative.add(derivative);
        }
        return oldDerivative;
    }
}
