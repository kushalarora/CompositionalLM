package com.kushalarora.compositionalLM.derivatives;

import com.kushalarora.compositionalLM.optimizer.IIndexed;
import com.kushalarora.compositionalLM.optimizer.IIndexedSized;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.AdaGrad;

import java.io.Serializable;
import java.util.List;

/**
 * Created by karora on 6/30/15.
 */

@Slf4j
public abstract class AbstractBaseDerivativeClass<T extends IIndexedSized> implements Serializable {
    protected AdaGrad adaGrad;
    protected T data;
    protected int[] shape;

    public AbstractBaseDerivativeClass(int[] shape, T data) {
        adaGrad = new AdaGrad(shape);
        this.data = data;
        this.shape = shape;
    }

    /**
     * Sum all entries, if there is a Nan or Inf result will
     * not be a finite number.
     */
    protected boolean containsNanOrInf(INDArray arr) {
        double sum = arr.sum(Integer.MAX_VALUE).getDouble();
        return !Double.isFinite(sum);
    }

    protected INDArray clampDerivativeIfNeeded(INDArray arr) {
        int[] arrShape = arr.shape();
        INDArray arr2 = arr.linearView();
        for (int i = 0; i < arr2.shape()[0]; i++) {
            double absValue = Math.abs(arr2.getDouble(i));
            if (absValue > 10) {
                arr2.putScalar(i, 10 * arr2.getDouble(i)/absValue);
            }
        }

        return arr2.reshape(arrShape);
    }
}
