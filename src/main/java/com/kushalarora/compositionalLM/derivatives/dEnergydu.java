package com.kushalarora.compositionalLM.derivatives;

import com.kushalarora.compositionalLM.model.CompositionalGrammar;
import com.kushalarora.compositionalLM.model.Model;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by karora on 6/21/15.
 */
class dEnergydu extends AbstractBaseDerivativeClass implements IDerivative {
    INDArray dEdu;

    public dEnergydu(Model model) {
        super(model);
        dEdu = Nd4j.zeros(model.params.getDimensions());
    }

    public void clear() {
        // Wipe clean
        for (int i = 0; i < model.params.getDimensions(); i++) {
            dEdu.putScalar(i, 0);
        }
    }

    public INDArray calcDerivative(CompositionalGrammar.CompositionalInsideOutsideScorer scorer) {
        int length = scorer.getCurrentSentence().size();
        INDArray[][][] compositionMatrix = scorer.getCompositionMatrix();
        float[][][] compositionMu = scorer.getMuScore();

        // do leaf nodes
        for (int start = 0; start < length; start++) {
            int end = start + 1;
            int split = start;

            INDArray compositionVector = compositionMatrix[start][end][split];
            float dE = model.energyDerivative(compositionVector);
            dEdu.add(compositionVector.muli(
                    dE * compositionMu[start][end][split]));
        }

        for (int diff = 2; diff <= length; diff++) {
            for (int start = 0; start + diff < length; start++) {
                int end = start + diff;
                for (int split = start + 1; split < end; split++) {
                    INDArray compositionVector = compositionMatrix[start][end][split];
                    float dE = model.energyDerivative(compositionVector);
                    dEdu = dEdu.add(
                            compositionVector.muli(
                                    dE * compositionMu[start][end][split]));
                }
            }
        }
        return dEdu;
    }
}