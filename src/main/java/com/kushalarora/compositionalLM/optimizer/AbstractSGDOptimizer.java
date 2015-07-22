package com.kushalarora.compositionalLM.optimizer;

import com.kushalarora.compositionalLM.model.AbstractDerivatives;
import com.kushalarora.compositionalLM.model.IParameter;
import com.kushalarora.compositionalLM.model.IParameterDerivatives;
import com.kushalarora.compositionalLM.options.Options;
import lombok.extern.slf4j.Slf4j;

import java.util.List;

/**
 * Created by karora on 7/14/15.
 */

@Slf4j
public abstract class AbstractSGDOptimizer<T extends IIndexed> extends AbstractOptimizer<T> {

    private int count;

    protected AbstractSGDOptimizer(Options op) {
        super(op);
        count = 0;
    }

    public void updateParams(IParameterDerivatives<T> derivatives) {
        getParams().update(
                derivatives.mul(
                        -1 * op.trainOp.learningRate / count));
    }

    public void calcLearningRate(final IParameterDerivatives<T> derivatives) {
        count++;
    }

    public void clearLearningRate() {
        count = 0;
    }
}
