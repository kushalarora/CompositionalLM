package com.kushalarora.compositionalLM.model;

import java.io.Serializable;

/**
 * Created by karora on 7/14/15.
 */
public interface IParameter extends Serializable {
    public void update(IParameterDerivatives derivatives);
}
