package com.kushalarora.compositionalLM.optimizer;

import com.kushalarora.compositionalLM.model.IParameter;
import com.kushalarora.compositionalLM.model.IParameterDerivatives;

import java.util.List;

/**
 * Created by karora on 7/7/15.
 */
public interface IOptimizer<T> {

    public IParameterDerivatives calcDerivative(final T sample);

    public void updateParams(final IParameterDerivatives<T> derivatives);

    public IParameter getParams();

    public void fitRoutine(int startIdx, List<T> data);

    public void derivativeAccumulator(IParameterDerivatives<T> derivatives);

    public IParameterDerivatives getAccumulatedDerivative();

    public void flushDerivaiveAccumulator();

    public void calcLearningRate(final T sample, final IParameterDerivatives<T> derivatives);
}
