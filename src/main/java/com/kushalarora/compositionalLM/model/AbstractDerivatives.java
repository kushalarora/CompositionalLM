package com.kushalarora.compositionalLM.model;

import com.kushalarora.compositionalLM.derivatives.IDerivatives;
import com.kushalarora.compositionalLM.optimizer.IIndexed;
import lombok.Getter;

/**
 * Created by karora on 7/22/15.
 */
public abstract class AbstractDerivatives<T extends IIndexed> implements IDerivatives<T> {
    protected T data;
    @Getter
    protected double score;
    public AbstractDerivatives(T data) {
        this.data = data;
    }
    public T getData() {
        return data;
    }
}
