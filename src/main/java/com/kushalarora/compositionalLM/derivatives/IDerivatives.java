package com.kushalarora.compositionalLM.derivatives;

import com.kushalarora.compositionalLM.model.CompositionalInsideOutsideScore;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.optimizer.IIndexedSized;

/**
 * Created by karora on 7/14/15.
 */
public interface IDerivatives<T extends IIndexedSized> extends Cloneable {
    public void add(IDerivatives<T> derivatives);

    public void mul(double learningRate);

    public void calcDerivative();

    public void clear();

    public T getData();

    public IDerivatives<T> adaGrad(IDerivatives<T> derivatives);

    public double getScore();
}
