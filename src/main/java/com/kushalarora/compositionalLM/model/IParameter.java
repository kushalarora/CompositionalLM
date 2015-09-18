package com.kushalarora.compositionalLM.model;

import com.kushalarora.compositionalLM.derivatives.IDerivatives;

import java.io.Serializable;

/**
 * Created by karora on 7/14/15.
 */
public interface IParameter<T> extends Serializable {
    public void update(IDerivatives<T> derivatives);
}
