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
public class dXdXw extends AbstractBaseDerivativeClass {
    INDArray[][][][] dXdXw;

    public dXdXw(Model model) {
        super(model);
        int dim = model.params.getDimensions();
        int V = model.params.getVocabSize();
        dXdXw = new INDArray[V][][][];
    }

    public INDArray[][][][] calcDerivative(CompositionalGrammar.CompositionalInsideOutsideScorer scorer) {
        List<Word> sentence = scorer.getCurrentSentence();
        int length = sentence.size();

        int[] indexes = new int[length];
        for (int i = 0; i < length; i++) {
            indexes[i] = sentence.get(i).getIndex();
        }
        int dim = model.params.getDimensions();
        int V = model.params.getVocabSize();

        for (int v = 0; v < V; v++) {
            dXdXw[v] = new INDArray[length][][];
            for (int start = 0; start < length; start++) {
                dXdXw[v][start] = new INDArray[length + 1][];
                for (int end = start; end <= length; end++) {
                    dXdXw[v][start][end] = new INDArray[length];
                }
            }
        }

        INDArray[][] dXdXwi = new INDArray[length][length + 1];

        INDArray[][] phraseMatrix = scorer.getPhraseMatrix();
        float[][][] compositionISplitScore = scorer.getCompositionISplitScore();
        float[][] compositionIScore = scorer.getInsideSpanProb();


        for (int i = 0; i < length; i++) {
            int index = indexes[i];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    for (int d = 0; d < dim; d++) {
                        dXdXwi[start][end].putScalar(d, 0);
                    }
                }
            }

            // dc/dc = 1 and dc/dx = 0 if c != x
            dXdXwi[i][i + 1] = Nd4j.eye(dim);

            for (int diff = 2; diff <= length; diff++) {
                for (int start = 0; start + diff <= length; start++) {
                    int end = start + diff;
                    for (int split = start + 1; split < end; split++) {

                        // Calculate f'(c_1, c_2)
                        INDArray child1 = phraseMatrix[start][split];
                        INDArray child2 = phraseMatrix[split][end];
                        INDArray dC = model.composeDerivative(child1, child2);

                        // [dc_1dW_ij dc_2dW_ij].transpose()
                        INDArray dC12 = Nd4j.concat(0, dXdXwi[start][split], dXdXwi[split][end]);

                        dXdXw[index][start][end][split] =
                                // f'(c1, c2) \dot
                                dC.muli(
                                        // W *
                                        model.params.getW().mul(
                                                // [dc_1 dc_2]^T)) *
                                                dC12));

                        // weighted marginalization over split
                        dXdXwi[start][end] = dXdXwi[start][end].add(
                                dXdXw[index][start][end][split].muli(
                                        // \pi[start][end][split]
                                        compositionISplitScore[start][end][split]));
                    }

                    // dXdXwi /= \pi[start][end]
                    dXdXwi[start][end] = dXdXwi[start][end].divi(
                            compositionIScore[start][end]);
                }
            }
        }
        return dXdXw;
    }

    public void clear() {

    }
}
