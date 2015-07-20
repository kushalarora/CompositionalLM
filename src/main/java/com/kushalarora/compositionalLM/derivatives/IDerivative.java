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

    public IDerivative add(IDerivative other);

    public IDerivative mul(double learningRate);
}


