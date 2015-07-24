package com.kushalarora.compositionalLM.derivatives;

import com.kushalarora.compositionalLM.lang.Word;
import com.kushalarora.compositionalLM.model.CompositionalGrammar;
import com.kushalarora.compositionalLM.model.Model;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.AdaGrad;

import java.io.Serializable;
import java.util.List;

/**
 * Created by karora on 6/30/15.
 */
public abstract class AbstractBaseDerivativeClass implements Serializable {
    protected final Model model;
    protected AdaGrad adaGrad;

    public AbstractBaseDerivativeClass(Model model, int[] shape) {
        adaGrad = new AdaGrad(shape);
        this.model = model;
    }

    public AbstractBaseDerivativeClass(Model model) {
        this.model = model;
    }

    protected boolean containsNanOrInf(INDArray arr) {
        double sum = arr.sum(Integer.MAX_VALUE).getDouble();
        return !Double.isFinite(sum);
    }
}
