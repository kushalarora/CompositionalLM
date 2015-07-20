package com.kushalarora.compositionalLM.model;

import com.kushalarora.compositionalLM.derivatives.dQdW;
import com.kushalarora.compositionalLM.derivatives.dQdXw;
import com.kushalarora.compositionalLM.derivatives.dQdu;
import com.kushalarora.compositionalLM.lang.Word;
import lombok.Getter;

import java.util.List;

/**
 * Created by karora on 7/14/15.
 */

@Getter
public class Derivatives implements IParameterDerivatives<List<Word>> {
    private dQdW dqdw;
    private dQdu dqdu;
    private dQdXw dqdxw;
    private CompositionalGrammar.CompositionalInsideOutsideScorer scorer;


    public Derivatives(Model model, CompositionalGrammar.CompositionalInsideOutsideScorer scorer) {
        // IMPORTANT::The order must be preserved here
        // all derivatives should be the last one to be
        // initialized
        dqdu = new dQdu(model);
        dqdw = new dQdW(model);
        dqdxw = new dQdXw(model);
        this.scorer = scorer;
    }

    public IParameterDerivatives add(IParameterDerivatives derivatives) {
        dqdu = (dQdu) dqdu.add(((Derivatives) derivatives).dqdu);
        dqdw = (dQdW) dqdw.add(((Derivatives) derivatives).dqdw);
        dqdxw = (dQdXw) dqdxw.add(((Derivatives) derivatives).dqdxw);
        return this;
    }

    public IParameterDerivatives mul(double learningRate) {
        dqdu = (dQdu) dqdu.mul(learningRate);
        dqdw = (dQdW) dqdw.mul(learningRate);
        dqdxw = (dQdXw) dqdxw.mul(learningRate);
        return this;
    }

    public IParameterDerivatives clear() {
        dqdu.clear();;
        dqdw.clear();
        dqdxw.clear();
        return this;
    }

    public IParameterDerivatives<List<Word>> calcDerivative(List<Word> data) {
        clear();
        scorer.computeCompInsideOutsideScores(data);
        dqdu.calcDerivative(data, scorer);
        dqdw.calcDerivative(data, scorer);
        dqdxw.calcDerivative(data, scorer);
        return this;
    }
}