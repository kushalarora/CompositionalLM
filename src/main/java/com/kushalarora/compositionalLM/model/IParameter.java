package com.kushalarora.compositionalLM.model;

import java.io.Serializable;

/**
 * Created by karora on 7/14/15.
 */
public interface IParameter<T> extends Serializable {
    public void update(IParameterDerivatives<T> derivatives);
}
