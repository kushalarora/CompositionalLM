package com.kushalarora.compositionalLM.derivatives;

import com.kushalarora.compositionalLM.lang.Sentence;
import com.kushalarora.compositionalLM.lang.StanfordCompositionalInsideOutsideScore;
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
    private StanfordCompositionalInsideOutsideScore score;
    private Model model;
    public Derivatives(Options op, Model model, StanfordCompositionalInsideOutsideScore score) {
        super(score.getSentence());
        // IMPORTANT::The order must be preserved here
        // all derivatives should be the last one to be
        // initialized
        this.model = model;
        this.score = score;
        dqdu = new dQdu(model.getDimensions(), data, op);
        dqdw = new dQdW(model.getDimensions(), data, op);
        dqdxw = new dQdXw(model.getDimensions(), model.getVocabSize(), data, op);
        this.op = op;
    }

    public Derivatives(Options op, Model model, dQdW dqdw, dQdu dqdu, dQdXw dqdxw) {
        super(new Sentence(-1));
        this.model = model;
        this.dqdw = dqdw;
        this.dqdu = dqdu;
        this.dqdxw = dqdxw;
        this.op = op;
    }

    /**
     * This is just used for accumulation
     * */
    public Derivatives(Model model, Options op) {
        super(new Sentence(-1));
        // IMPORTANT::The order must be preserved here
        // all derivatives should be the last one to be
        // initialized
        this.model = model;
        dqdu = new dQdu(model.getDimensions(), new Sentence(-1), op);
        dqdw = new dQdW(model.getDimensions(), new Sentence(-1), op);
        dqdxw = new dQdXw(model.getDimensions(), model.getVocabSize(), new Sentence(-1), op);
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
    calcDerivative() {
        int idx = data.getIndex();
        int sz = data.size();
        dqdu.calcDerivative(model, score);
        dqdw.calcDerivative(model, score);
        dqdxw.calcDerivative(model, score);

        log.info("dQdu Norm2:{}(len={}) = {}", idx, sz, dqdu.norm());
        log.info("dQdW Norm2:{}(len={}) = {}", idx, sz, dqdw.norm());
        log.info("dQdXw Norm2:{}(len={}) = {}", idx, sz, dqdxw.norm());

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
        return new Derivatives(op, model,
                (dQdW) dqdw.adaGrad(
                        ((Derivatives) derivatives).dqdw),
                (dQdu) dqdu.adaGrad(
                        ((Derivatives) derivatives).dqdu),
                (dQdXw) dqdxw.adaGrad(
                        ((Derivatives) derivatives).dqdxw)
        );
    }

    public double getScore() {
        return score.getSentenceScore();
    }
}