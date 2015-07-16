package com.kushalarora.compositionalLM.optimizer;

import com.kushalarora.compositionalLM.derivatives.IDerivative;
import com.kushalarora.compositionalLM.model.IParameter;
import com.kushalarora.compositionalLM.model.IParameterDerivatives;

/**
 * Created by karora on 7/7/15.
 */
public interface IOptimizer<T> {

    public IParameterDerivatives calcDerivative(T sample);

    public void updateParams(IParameter parameter);

    public IParameter getParams();
}
