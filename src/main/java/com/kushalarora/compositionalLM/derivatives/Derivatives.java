package com.kushalarora.compositionalLM.derivatives;

import com.kushalarora.compositionalLM.lang.Sentence;
import com.kushalarora.compositionalLM.lang.StanfordCompositionalInsideOutsideScore;
import com.kushalarora.compositionalLM.model.AbstractDerivatives;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.options.Options;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;

import static com.kushalarora.compositionalLM.utils.ObjectSizeFetcher.getSize;

@Getter
@Slf4j
public class Derivatives extends AbstractDerivatives<Sentence> {
    private  Options op;
    private dQdu dqdu;
    private dQdh1 dqdh1;
    private dQdh2 dqdh2;
    private dQdW dqdw;
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
        dqdh1 = new dQdh1(model.getDimensions(), data, op);
        dqdh2 = new dQdh2(model.getDimensions(), data, op);
        dqdw = new dQdW(model.getDimensions(), data, op);
        dqdxw = new dQdXw(model.getDimensions(), model.getParams().getGrammarVocabSize(), data, op);
        this.op = op;
    }

    public Derivatives(Options op, Model model, dQdW dqdw, dQdu dqdu,
                       dQdh1 dqdh1, dQdh2 dqdh2,  dQdXw dqdxw) {
        super(new Sentence(-1));
        this.model = model;
        this.dqdu = dqdu;
        this.dqdh1 = dqdh1;
        this.dqdh2 = dqdh2;
        this.dqdw = dqdw;
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
        dqdh1 = new dQdh1(model.getDimensions(), new Sentence(-1), op);
        dqdh2 = new dQdh2(model.getDimensions(), new Sentence(-1), op);
        dqdw = new dQdW(model.getDimensions(), new Sentence(-1), op);
        dqdxw = new dQdXw(model.getDimensions(), model.getParams().getGrammarVocabSize(), new Sentence(-1), op);
    }

    public void add(IDerivatives derivatives) {
        Derivatives dv = (Derivatives) derivatives;
        if (dv.containsNanOrInf()) {
            log.error("Inf or Nan present in derivative in {}. Ignoring", dv.getData());
            return;
        }
        dqdu.add(dv.dqdu);
        dqdh1.add(dv.dqdh1);
        dqdh2.add(dv.dqdh2);
        dqdw.add(dv.dqdw);
        dqdxw.add(dv.dqdxw);
    }


    public void mul(double learningRate) {
        dqdu.mul(learningRate);
        dqdh1.mul(learningRate);
        dqdh2.mul(learningRate);
        dqdw.mul(learningRate);
        dqdxw.mul(learningRate);
    }


    public void clear() {
        dqdu.clear();
        dqdh1.clear();
        dqdh2.clear();
        dqdw.clear();
        dqdxw.clear();
    }

    public void
    calcDerivative() {
        int idx = data.getIndex();
        int sz = data.size();
        dqdu.calcDerivative(model, score);
        dqdh1.calcDerivative(model, score);
        dqdh2.calcDerivative(model, score);
        dqdw.calcDerivative(model, score);
        dqdxw.calcDerivative(model, score);

        if (op.debug) {
            log.info("dQdu Norm2:{}(len={}) = {}", idx, sz, dqdu.norm());
            log.info("dQdh1 Norm2:{}(len={}) = {}", idx, sz, dqdh1.norm());
            log.info("dQdh2 Norm2:{}(len={}) = {}", idx, sz, dqdh2.norm());
            log.info("dQdW Norm2:{}(len={}) = {}", idx, sz, dqdw.norm());
            log.info("dQdXw Norm2:{}(len={}) = {}", idx, sz, dqdxw.norm());

            log.info("Memory Size Derivatives: {}:: {}\n" +
                            "\t {} => {}  MB\n" +
                            "\t {} => {}  MB\n" +
                            "\t {} => {}  MB\n" +
                            "\t {} => {} MB\n" +
                            "\t {} => {} MB\n" +
                            "total => {} MB",
                    idx, sz,
                    "dQdu", getSize(dqdu),
                    "dQdh1", getSize(dqdh1),
                    "dQdh2", getSize(dqdh2),
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
                dqdh1.containsNanOrInf() ||
                dqdh2.containsNanOrInf() ||
                dqdw.containsNanOrInf() ||
                dqdxw.containsNanOrInf();
    }

    public Derivatives adaGrad(IDerivatives<Sentence> derivatives) {
        return new Derivatives(op, model,
                (dQdW) dqdw.adaGrad(
                        ((Derivatives) derivatives).dqdw),
                (dQdu) dqdu.adaGrad(
                        ((Derivatives) derivatives).dqdu),
                (dQdh1) dqdh1.adaGrad(
                        ((Derivatives) derivatives).dqdh1),
                (dQdh2) dqdh2.adaGrad(
                        ((Derivatives) derivatives).dqdh2),
                (dQdXw) dqdxw.adaGrad(
                        ((Derivatives) derivatives).dqdxw)
        );
    }

    public double getScore() {
        return score.getSentenceScore();
    }

    public static void preProcessOnBatch() {
       // do nothing
    }

    public static void postProcessOnBatch() {
        dQdu.cleanZLeaf();
        dQdXw.cleanZLeaf();
    }
}
