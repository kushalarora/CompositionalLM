package com.kushalarora.compositionalLM.model;

import com.kushalarora.compositionalLM.derivatives.dQdW;
import com.kushalarora.compositionalLM.derivatives.dQdXw;
import com.kushalarora.compositionalLM.derivatives.dQdu;
import com.kushalarora.compositionalLM.lang.Sentence;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.factory.Nd4j;

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


    public void clear() {
        dqdu.clear();
        dqdw.clear();
        dqdxw.clear();
    }

    public void
    calcDerivative(CompositionalGrammar
                           .CompositionalInsideOutsideScore scorer) {
        dqdu.calcDerivative(data, scorer);
        log.info("dQdu Norm2:{}(len={}) = {}", data.getIndex(), data.size(), Nd4j.norm2(dqdu.getDQdu()));
        dqdw.calcDerivative(data, scorer);
        log.info("dQdW Norm2:{}(len={}) = {}", data.getIndex(), data.size(), Nd4j.norm2(dqdw.getDQdW()));
        dqdxw.calcDerivative(data, scorer);
        log.info("dQdXw Norm2:{}(len={}) = {}", data.getIndex(), data.size(), Nd4j.norm2(dqdxw.getDQdXw()));
    }

    public Sentence getData() {
        return data;
    }

    private boolean containsNanOrInf() {
        return dqdu.containsNanOrInf() ||
                dqdw.containsNanOrInf() ||
                dqdxw.containsNanOrInf();
    }

    public Derivatives adaGrad(IDerivatives<Sentence> derivatives) {
        return new Derivatives(data,
                (dQdW) dqdw.adaGrad(
                        ((Derivatives) derivatives).dqdw),
                (dQdu) dqdu.adaGrad(
                        ((Derivatives) derivatives).dqdu),
                (dQdXw) dqdxw.adaGrad(
                        ((Derivatives) derivatives).dqdxw)
        );
    }
}