package com.kushalarora.compositionalLM.derivatives;

import com.kushalarora.compositionalLM.lang.Word;
import com.kushalarora.compositionalLM.model.CompositionalGrammar;
import com.kushalarora.compositionalLM.model.Model;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

/**
 * Created by karora on 6/21/15.
 */
public class dQdXw extends AbstractBaseDerivativeClass implements IDerivative {
    private dXdXw dxdxw;
    private INDArray dEdXw;


    public dQdXw(Model model, dXdXw dxdxw) {
        super(model);
        this.dxdxw = dxdxw;
        int dim = model.params.getDimensions();
        int V = model.params.getVocabSize();
        dEdXw = Nd4j.zeros(V, dim);
    }

    public INDArray calcDerivative(CompositionalGrammar.CompositionalInsideOutsideScorer scorer) {

        List<Word> sentence = scorer.getCurrentSentence();
        int length = sentence.size();

        int[] indexes = new int[length];
        for (int i = 0; i < length; i++) {
            indexes[i] = sentence.get(i).getIndex();
        }

        INDArray[][][][] dxdxwArr = dxdxw.calcDerivative(scorer);
        INDArray[][][] compositionMatrix = scorer.getCompositionMatrix();
        float[][][] compositionalMu = scorer.getMuScore();
        INDArray[][] phraseMatrix = scorer.getPhraseMatrix();

        int V = model.params.getVocabSize();
        int dim = model.params.getDimensions();

        for (int i = 0; i < length; i++) {
            int index = indexes[i];
            INDArray dEdXw_i = Nd4j.zeros(dim);

            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    for (int split = start + 1; split < end; split++) {
                        float dE = model.energyDerivative(compositionMatrix[start][end][split],
                                phraseMatrix[start][split], phraseMatrix[split][end]);

                        INDArray udXdWArr = model.params.getU().mmul(
                                dxdxwArr[i][start][end][split]);

                        dEdXw_i.add(
                                udXdWArr.muli(
                                        compositionalMu[start][end][split]));
                    }
                }
            }

            dEdXw.put(index, dEdXw_i);
        }
        return dEdXw;
    }

    public void clear() {

    }
}
