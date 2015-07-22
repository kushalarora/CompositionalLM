package com.kushalarora.compositionalLM.model;

import com.kushalarora.compositionalLM.derivatives.dQdW;
import com.kushalarora.compositionalLM.derivatives.dQdXw;
import com.kushalarora.compositionalLM.derivatives.dQdu;
import com.kushalarora.compositionalLM.lang.Sentence;
import com.kushalarora.compositionalLM.lang.Word;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;

import java.util.List;

/**
 * Created by karora on 7/14/15.
 */

@Getter
@Slf4j
public class Derivatives extends AbstractDerivatives<Sentence> {
    private dQdW dqdw;
    private dQdu dqdu;
    private dQdXw dqdxw;

    public Derivatives(Model model, Sentence sentence) {
        super(sentence);
        // IMPORTANT::The order must be preserved here
        // all derivatives should be the last one to be
        // initialized
        dqdu = new dQdu(model);
        dqdw = new dQdW(model);
        dqdxw = new dQdXw(model);
    }


    /**
     * This is just used for accumulation
     * @param model
     */
    public Derivatives(Model model) {
        super(new Sentence(-1));
        // IMPORTANT::The order must be preserved here
        // all derivatives should be the last one to be
        // initialized
        dqdu = new dQdu(model);
        dqdw = new dQdW(model);
        dqdxw = new dQdXw(model);
    }

    public IParameterDerivatives<Sentence> add(IParameterDerivatives derivatives) {
        Derivatives dv = (Derivatives) derivatives;
        if (dv.containsNanOrInf()) {
            log.error("Inf or Nan present in derivative in {}. Ignoring", data);
            return this;
        }
        dqdu = (dQdu) dqdu.add(dv.dqdu);
        dqdw = (dQdW) dqdw.add(dv.dqdw);
        dqdxw = (dQdXw) dqdxw.add(dv.dqdxw);
        return this;
    }


    public IParameterDerivatives<Sentence> mul(double learningRate) {
        dqdu = (dQdu) dqdu.mul(learningRate);
        dqdw = (dQdW) dqdw.mul(learningRate);
        dqdxw = (dQdXw) dqdxw.mul(learningRate);
        return this;
    }

    public void clear() {
        dqdu.clear();
        dqdw.clear();
        dqdxw.clear();
    }

    public IParameterDerivatives<Sentence>
    calcDerivative(CompositionalGrammar
            .CompositionalInsideOutsideScore scorer) {
        dqdu.calcDerivative(data, scorer);
        dqdw.calcDerivative(data, scorer);
        dqdxw.calcDerivative(data, scorer);
        return this;
    }

    public Sentence getData() {
        return data;
    }


    private boolean containsNanOrInf() {
        return dqdu.containsNanOrInf() ||
                dqdw.containsNanOrInf() ||
                dqdxw.containsNanOrInf();
    }
}