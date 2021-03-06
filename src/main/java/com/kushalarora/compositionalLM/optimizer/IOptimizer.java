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

    public double getValidationScore(T data);

    public double getTrainScore(T data);

    public void saveModel(int iter, int epoch);
}
