package com.kushalarora.compositionalLM.derivatives;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.AdaGrad;

import java.io.Serializable;

/**
 * Created by karora on 6/30/15.
 */
public abstract class AbstractBaseDerivativeClass implements Serializable {
    protected AdaGrad adaGrad;

    public AbstractBaseDerivativeClass(int[] shape) {
        adaGrad = new AdaGrad(shape);
    }

    /**
     * Sum all entries, if there is a Nan or Inf result will
     * not be a finite number.
     */
    protected boolean containsNanOrInf(INDArray arr) {
        double sum = arr.sum(Integer.MAX_VALUE).getDouble();
        return !Double.isFinite(sum) || (sum <= -100) || (sum >= 100);
    }
}
