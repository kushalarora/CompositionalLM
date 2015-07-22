package com.kushalarora.compositionalLM.optimizer;

import com.kushalarora.compositionalLM.model.AbstractDerivatives;
import com.kushalarora.compositionalLM.model.IParameter;
import com.kushalarora.compositionalLM.model.AbstractDerivatives;
import com.kushalarora.compositionalLM.model.IParameterDerivatives;

import java.util.List;

/**
 * Created by karora on 7/7/15.
 */
public interface IOptimizer<T extends IIndexed> {

    public IParameterDerivatives<T> calcDerivative(final T sample);

    public void updateParams(final IParameterDerivatives<T> derivatives);

    public IParameter getParams();

    public void fitRoutine(List<T> data);

    public void derivativeAccumulator(IParameterDerivatives<T> derivatives);

    public IParameterDerivatives<T> getAccumulatedDerivative();

    public void clearLearningRate();

    public void flushDerivaiveAccumulator();

    public void calcLearningRate(final IParameterDerivatives<T> derivatives);
}
