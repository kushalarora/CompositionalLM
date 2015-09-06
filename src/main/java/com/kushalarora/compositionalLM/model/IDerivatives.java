package com.kushalarora.compositionalLM.model;

import java.io.Serializable;

/**
 * Created by karora on 7/14/15.
 */
public interface IDerivatives<T> extends Cloneable {
    public void add(IDerivatives<T> derivatives);

    public void mul(double learningRate);

    public void calcDerivative(CompositionalGrammar.CompositionalInsideOutsideScore scorer);

    public void clear();

    public T getData();

    public IDerivatives<T> adaGrad(IDerivatives<T> derivatives);
}
