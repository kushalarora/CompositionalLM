package com.kushalarora.compositionalLM.derivatives;

import com.kushalarora.compositionalLM.model.CompositionalInsideOutsideScore;
import com.kushalarora.compositionalLM.model.Model;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

/**
 * Created by karora on 6/21/15.
 */
public interface IDerivative<T extends List> {
    public void calcDerivative(Model model, CompositionalInsideOutsideScore scorer);

    public void add(IDerivative<T> other);

    public void mul(double learningRate);

    public boolean containsNanOrInf();

    public IDerivative adaGrad(IDerivative gradient);
}


