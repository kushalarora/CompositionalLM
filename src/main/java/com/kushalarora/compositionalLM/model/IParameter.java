package com.kushalarora.compositionalLM.model;

import com.kushalarora.compositionalLM.derivatives.IDerivatives;
import com.kushalarora.compositionalLM.optimizer.IIndexedSized;

import java.io.Serializable;

/**
 * Created by karora on 7/14/15.
 */
public interface IParameter<T extends IIndexedSized> extends Serializable {
    public void update(IDerivatives<T> derivatives);
}
