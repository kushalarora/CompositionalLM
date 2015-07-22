package com.kushalarora.compositionalLM.model;

import com.kushalarora.compositionalLM.optimizer.IIndexed;

/**
 * Created by karora on 7/22/15.
 */
public abstract class AbstractDerivatives<T extends IIndexed> implements IParameterDerivatives<T> {
    protected T data;
    public AbstractDerivatives(T data) {
        this.data = data;
    }
    public T getData() {
        return data;
    }
}
