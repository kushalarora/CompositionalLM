package com.kushalarora.compositionalLM.model;

import java.io.Serializable;

/**
 * Created by karora on 7/14/15.
 */
public interface IParameterDerivatives extends Serializable {
    public IParameterDerivatives add(IParameterDerivatives derivatives);
    public IParameterDerivatives mul(double learningRate);
}
