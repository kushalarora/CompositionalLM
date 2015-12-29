package com.kushalarora.compositionalLM.derivatives;

import com.kushalarora.compositionalLM.optimizer.IIndexed;
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
public abstract class AbstractBaseDerivativeClass<T extends List<? extends IIndexed>> implements Serializable {
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
        double norm2 = Nd4j.norm2(arr).getDouble(0);
        return !Double.isFinite(sum) || (norm2 > 10000);
    }

    protected INDArray clampDerivativeIfNeeded(INDArray arr) {
        double norm2 = Nd4j.norm2(arr).getDouble(0);
        if ((norm2 > 100)) {
            log.error("Clipping gradiant of shape {} for data: {}. Norm was {}", shape, data, norm2);
            return arr.div(norm2).mul(100);
        }
        return arr;
    }
}
