package com.kushalarora.compositionalLM.derivatives;

import com.kushalarora.compositionalLM.lang.Word;
import com.kushalarora.compositionalLM.model.CompositionalGrammar;
import com.kushalarora.compositionalLM.model.Model;
import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

/**
 * Created by karora on 6/21/15.
 * Energy function E is given as
 * E = g(u^Tp) where p is phrase vector..
 * dEdu = g'(u.t().dot(p))p
 *
 * dQdu = \sum{start}{end}{split} dEdu(start, end, split) * \mu(start, end, split)
 */
public class dQdu extends AbstractBaseDerivativeClass implements IDerivative {
    @Getter
    private INDArray dQdu;

    public dQdu(Model model) {
        super(model);
        dQdu = Nd4j.zeros(model.getDimensions());
    }

    public dQdu(dQdu dqdu) {
        super(dqdu.model);
        dQdu = dqdu.dQdu;
    }

    public void clear() {
        // Wipe cleaern
        for (int i = 0; i < model.getDimensions(); i++) {
            dQdu.putScalar(i, 0);
        }
    }

    public IDerivative add(IDerivative other) {
        dQdu  = dQdu.add(((dQdu)other).getDQdu());
        return this;
    }

    public IDerivative mul(double learningRate) {
        dQdu = dQdu.mul(learningRate);
        return this;
    }

    public INDArray calcDerivative(List<Word> sentence, CompositionalGrammar.CompositionalInsideOutsideScorer scorer) {
        int length = sentence.size();
        INDArray[][][] compositionMatrix = scorer.getCompositionMatrix();
        INDArray[][] phraseMatrix = scorer.getPhraseMatrix();
        float[][][] compositionMu = scorer.getMuScore();
        float[][] compositionalIScore = scorer.getInsideSpanProb();

        // do leaf nodes
        for (int start = 0; start < length; start++) {
            int end = start + 1;
            int split = start;

            // For leaf nodes we consider the phrase
            INDArray phraseVector = phraseMatrix[start][end];

            // dE = g'(u.t().dot(p))
            float dE = model.energyDerivative(phraseVector);

            // dEdu = dE * p = g'(u.t().dot(p)) * p
            INDArray dEdu = phraseVector.muli(dE);

            // dQdu += dEdu * \mu[start][end][split]
            dQdu = dQdu.add(
                    dEdu.muli(
                            compositionMu[start][end][split]));
        }

        for (int diff = 2; diff <= length; diff++) {
            for (int start = 0; start + diff < length; start++) {
                int end = start + diff;
                for (int split = start + 1; split < end; split++) {

                    // Composition vector is parent's(start, end) embedding generated by
                    // child1 (start, split) and child2 (split, end)
                    INDArray compositionVector = compositionMatrix[start][end][split];

                    // dE = g'(u.t().dot(p))
                    float dE = model.energyDerivative(compositionVector);

                    // dEdu = dE * p = g'(u.t().dot(p)) * p
                    INDArray dEdu = compositionVector.muli(dE);

                    // dQdu += dEdu * \mu[start][end][split]
                    dQdu = dQdu.add(
                            dEdu.muli(
                                    compositionMu[start][end][split]));
                }
            }
        }
        dQdu = dQdu.div(compositionalIScore[0][length]);
        return dQdu;
    }
}