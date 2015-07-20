package com.kushalarora.compositionalLM.optimizer;

import com.kushalarora.compositionalLM.model.IParameter;
import com.kushalarora.compositionalLM.model.IParameterDerivatives;

import java.util.List;

/**
 * Created by karora on 7/7/15.
 */
public interface IOptimizer<T> {

    public IParameterDerivatives calcDerivative(T sample);

    public void updateParams(IParameterDerivatives derivatives);

    public IParameter getParams();

    public void fitRoutine(int startIdx, List<T> data);

    public void derivativeAccumulator(IParameterDerivatives derivatives);

    public IParameterDerivatives getAccumulatedDerivative();

    public void flushDerivaiveAccumulator();
}
