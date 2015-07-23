package com.kushalarora.compositionalLM.derivatives;

import com.kushalarora.compositionalLM.lang.Word;
import com.kushalarora.compositionalLM.model.CompositionalGrammar;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

/**
 * Created by karora on 6/21/15.
 */
public interface IDerivative {
    public INDArray calcDerivative(List<Word> sentence, CompositionalGrammar.CompositionalInsideOutsideScore scorer);

    public void clear();

    public void add(IDerivative other);

    public void mul(double learningRate);

    public boolean containsNanOrInf();

    public void mul(IDerivative adaGrad);

    public void power(double power);

    public void add(double bias);
}


