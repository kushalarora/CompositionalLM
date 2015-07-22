package com.kushalarora.compositionalLM.model;

import java.io.Serializable;

/**
 * Created by karora on 7/14/15.
 */
public interface IParameterDerivatives<T> extends Serializable {
    public IParameterDerivatives<T> add(IParameterDerivatives<T> derivatives);
    public IParameterDerivatives<T> mul(double learningRate);
    public IParameterDerivatives<T> calcDerivative(T data, CompositionalGrammar.CompositionalInsideOutsideScore scorer);
    public void clear();
    public T getData();
}
