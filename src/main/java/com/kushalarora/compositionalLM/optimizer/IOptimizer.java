package com.kushalarora.compositionalLM.optimizer;

import com.kushalarora.compositionalLM.model.IParameter;
import com.kushalarora.compositionalLM.model.IDerivatives;

import java.util.List;

/**
 * Created by karora on 7/7/15.
 */
public interface IOptimizer<T extends IIndexed, D extends IDerivatives<T>> {

    public D calcDerivative(final T sample);

    public void updateParams(final D derivatives);

    public IParameter getParams();

    public void derivativeAcc(D derivatives);

    public D getAccumulatedDerivative();

    public void clearLearningRate();

    public void flushDerivaiveAcc();

    public void calcLearningRate(final D derivatives);

    public double getValidationScore(T data);

    public void saveModel();
}
