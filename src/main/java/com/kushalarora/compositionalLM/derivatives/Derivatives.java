package com.kushalarora.compositionalLM.derivatives;

import com.kushalarora.compositionalLM.lang.Sentence;
import com.kushalarora.compositionalLM.model.AbstractDerivatives;
import com.kushalarora.compositionalLM.model.CompositionalInsideOutsideScore;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.options.Options;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.factory.Nd4j;

import static com.kushalarora.compositionalLM.utils.ObjectSizeFetcher.getSize;

/**
 * Created by karora on 7/14/15.
 */

@Getter
@Slf4j
public class Derivatives extends AbstractDerivatives<Sentence> {
    private  Options op;
    private dQdW dqdw;
    private dQdu dqdu;
    private dQdXw dqdxw;

    public Derivatives(Options op, int dimensions, int vocabSize, Sentence sentence) {
        super(sentence);
        // IMPORTANT::The order must be preserved here
        // all derivatives should be the last one to be
        // initialized
        dqdu = new dQdu(dimensions, sentence, op);
        dqdw = new dQdW(dimensions, sentence, op);
        dqdxw = new dQdXw(dimensions, vocabSize, sentence, op);
        this.op = op;
    }

    public Derivatives(Options op, Sentence sentence, dQdW dqdw, dQdu dqdu, dQdXw dqdxw) {
        super(sentence);
        this.dqdw = dqdw;
        this.dqdu = dqdu;
        this.dqdxw = dqdxw;
        this.op = op;
    }

    /**
     * This is just used for accumulation
     * */
    public Derivatives(int dimension, int vocabSize, Options op) {
        super(new Sentence(-1));
        // IMPORTANT::The order must be preserved here
        // all derivatives should be the last one to be
        // initialized
        dqdu = new dQdu(dimension, new Sentence(-1), op);
        dqdw = new dQdW(dimension, new Sentence(-1), op);
        dqdxw = new dQdXw(dimension, vocabSize, new Sentence(-1), op);
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
    calcDerivative(Model model, CompositionalInsideOutsideScore scorer) {
        int idx = data.getIndex();
        int sz = data.size();

        dqdu.calcDerivative(model, scorer);
        dqdw.calcDerivative(model, scorer);
        dqdxw.calcDerivative(model, scorer);

        log.info("dQdu Norm2:{}(len={}) = {}", idx, sz, dqdu.norm());
        log.info("dQdW Norm2:{}(len={}) = {}", idx, sz, dqdw.norm());
        log.info("dQdXw Norm2:{}(len={}) = {}", idx, sz, dqdxw.norm());
        score = scorer.getSentenceScore();

        if (op.debug) {
            log.info("Memory Size Derivatives: {}:: {}\n" +
                            "\t {} => {}  MB\n" +
                            "\t {} => {} MB\n" +
                            "\t {} => {} MB\n" +
                            "total => {} MB",
                    idx, sz,
                    "dQdu", getSize(dqdu),
                    "dQdW", getSize(dqdw),
                    "dqdxw", getSize(dqdxw),
                    getSize(this));
        }
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
        return new Derivatives(op, data,
                (dQdW) dqdw.adaGrad(
                        ((Derivatives) derivatives).dqdw),
                (dQdu) dqdu.adaGrad(
                        ((Derivatives) derivatives).dqdu),
                (dQdXw) dqdxw.adaGrad(
                        ((Derivatives) derivatives).dqdxw)
        );
    }
}