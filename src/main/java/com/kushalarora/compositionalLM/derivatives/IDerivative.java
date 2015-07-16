package com.kushalarora.compositionalLM.derivatives;

import com.kushalarora.compositionalLM.model.CompositionalGrammar;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by karora on 6/21/15.
 */
public interface IDerivative {
    public INDArray calcDerivative(CompositionalGrammar.CompositionalInsideOutsideScorer scorer);
    public void clear();
    public IDerivative add(IDerivative other);
    public IDerivative mul(double learningRate);
}


