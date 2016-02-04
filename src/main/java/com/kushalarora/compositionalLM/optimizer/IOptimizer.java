package com.kushalarora.compositionalLM.optimizer;

import com.kushalarora.compositionalLM.derivatives.IDerivatives;
import com.kushalarora.compositionalLM.model.IParameter;

/**
 * Created by karora on 7/7/15.
 */
public interface IOptimizer<T extends IIndexedSized, D extends IDerivatives> {

    public D calcDerivative(final T sample);

    public void updateParams(final D derivatives);

    public IParameter getParams();

    public void derivativeAcc(D derivatives);

    public D getAccumulatedDerivative();

    public void clearLearningRate();

    public void flushDerivaiveAcc();

    public void calcLearningRate(final D derivatives);

    public double getScore(T data);

    public void saveModel(int iter, int epoch);
}
