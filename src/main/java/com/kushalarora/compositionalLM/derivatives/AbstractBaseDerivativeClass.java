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
        adaGrad = new AdaGrad(shape, 0.1);
        this.data = data;
        this.shape = shape;
    }

    /**
     * Sum all entries, if there is a Nan or Inf result will
     * not be a finite number.
     */
    protected boolean containsNanOrInf(INDArray arr) {
        double sum = arr.sum(Integer.MAX_VALUE).getDouble(0, 0);
        return !Double.isFinite(sum);
    }

    protected INDArray clampDerivativeIfNeeded(INDArray arr) {
        double norm2 = Nd4j.norm2(arr).getDouble(0);
        double cutoff = 500;
        if (norm2 > cutoff) {
            log.error("Clipping gradiant of shape {} for data:{}::{}. Norm = {}",
                    shape, data.getIndex(), data.getSize(),  norm2);
            return arr.div(norm2).mul(cutoff);
        }
        return arr;
    }
}
