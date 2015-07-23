package com.kushalarora.compositionalLM.model;

import com.kushalarora.compositionalLM.derivatives.dQdW;
import com.kushalarora.compositionalLM.derivatives.dQdXw;
import com.kushalarora.compositionalLM.derivatives.dQdu;
import com.kushalarora.compositionalLM.lang.Sentence;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;

import static org.nd4j.linalg.ops.transforms.Transforms.pow;
import static org.nd4j.linalg.ops.transforms.Transforms.sqrt;

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

    public Derivatives(Sentence sentence, dQdW dqdw, dQdu dqdu, dQdXw dqdxw) {
        super(sentence);
        this.dqdw = dqdw;
        this.dqdu = dqdu;
        this.dqdxw = dqdxw;
    }


    /**
     * This is just used for accumulation
     *
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

    public void add(IDerivatives derivatives) {
        Derivatives dv = (Derivatives) derivatives;
        if (dv.containsNanOrInf()) {
            log.error("Inf or Nan present in derivative in {}. Ignoring", dv.getData());
            return;
        }
        dqdu.add(dv.dqdu);
        dqdw.add(dv.dqdw);
        dqdxw.add(dv.dqdxw);
    }


    public void mul(double learningRate) {
        dqdu.mul(learningRate);
        dqdw.mul(learningRate);
        dqdxw.mul(learningRate);
    }

    public void mul(IDerivatives<Sentence> adagrad) {
        dqdu.mul(((Derivatives) adagrad).getDqdu());
        dqdw.mul(((Derivatives) adagrad).getDqdw());
        dqdxw.mul(((Derivatives) adagrad).getDqdxw());
    }


    public void clear() {
        dqdu.clear();
        dqdw.clear();
        dqdxw.clear();
    }

    public void
    calcDerivative(CompositionalGrammar
                           .CompositionalInsideOutsideScore scorer) {
        dqdu.calcDerivative(data, scorer);
        dqdw.calcDerivative(data, scorer);
        dqdxw.calcDerivative(data, scorer);
    }

    public void power(double exponent) {
        dqdw.power(exponent);
        dqdu.power(exponent);
        dqdxw.power(exponent);
    }

    public Sentence getData() {
        return data;
    }


    private boolean containsNanOrInf() {
        return dqdu.containsNanOrInf() ||
                dqdw.containsNanOrInf() ||
                dqdxw.containsNanOrInf();
    }

    public IDerivatives<Sentence> deepcopy() {
        return new Derivatives(data,
                new dQdW(dqdw),
                new dQdu(dqdu),
                new dQdXw(dqdxw));
    }

    public void add(double bias) {
        dqdu.add(bias);
        dqdw.add(bias);
        dqdxw.add(bias);
    }
}