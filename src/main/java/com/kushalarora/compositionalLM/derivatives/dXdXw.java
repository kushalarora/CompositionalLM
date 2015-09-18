package com.kushalarora.compositionalLM.derivatives;

import com.kushalarora.compositionalLM.model.CompositionalInsideOutsideScore;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.optimizer.IIndexed;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

/**
 * Created by karora on 6/21/15.
 */

@Slf4j
public class dXdXw<T extends List<? extends IIndexed>> {
    @Getter
    private INDArray[][][][] dXdXw;
    @Getter
    private INDArray[][] dXdXwi;

    private int dim;
    private int V;
    private T data;
    private int length;

    public dXdXw(int dimension, int vocab, T data) {
        dim = dimension;
        V = vocab;
        this.data = data;
        length = data.size();
        dXdXw = new INDArray[V][][][];

        dXdXw = new INDArray[length][][][];
        dXdXwi = new INDArray[length][length + 1];

        for (int idx = 0; idx < length; idx++) {
            dXdXw[idx] = new INDArray[length][length + 1][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    dXdXw[idx][start][end] = new INDArray[length];
                    for (int split = start + 1; split < end; split++) {
                        dXdXw[idx][start][end][split] = Nd4j.create(dim, dim);
                    }
                }
            }
        }

        for (int start = 0; start < length; start++) {
            int end = start + 1;
            dXdXwi[start][end] = Nd4j.eye(dim);
        }

        for (int diff = 2; diff <= length; diff++) {
            for (int start = 0; start + diff <= length; start++) {
                int end = start + diff;
                dXdXwi[start][end] = Nd4j.zeros(dim, dim);
            }
        }
    }

    public INDArray[][][][] calcDerivative(Model model,
                                           CompositionalInsideOutsideScore scorer) {


        INDArray[][] phraseMatrix = scorer.getPhraseMatrix();
        double[][][] compositionISplitScore = scorer.getCompositionISplitScore();
        double[][] compositionIScore = scorer.getInsideSpanProb();

        for (int i = 0; i < length; i++) {

            // wipe dXdXwi arrays
            // dc/dc = 1 and dc/dx = 0 if c != x
            for (int start = 0; start < length; start++) {
                int end = start + 1;
                for (int d = 0; d < dim; d++) {
                    for (int d2 = 0; d2 < dim; d2++) {
                        dXdXwi[start][end].putScalar(
                                new int[]{d, d2}, d == d2 ? 1 : 0);
                    }
                }
            }

            for (int diff = 2; diff <= length; diff++) {
               for (int start = 0; start + diff <= length; start++) {
                   int end = start + diff;
                   for (int d = 0; d < dim; d++) {
                       for (int d2 = 0; d2 < dim; d2++) {
                           dXdXwi[start][end].putScalar(
                                   new int[]{d, d2}, 0);
                       }
                   }
               }
            }

            for (int diff = 2; diff <= length; diff++) {
                for (int start = 0; start + diff <= length; start++) {
                    int end = start + diff;
                    for (int split = start + 1; split < end; split++) {

                        // Calculate f'(c_1, c_2)
                        INDArray child1 = phraseMatrix[start][split];
                        INDArray child2 = phraseMatrix[split][end];
                        INDArray dC = model.composeDerivative(child1, child2);
                        dC = dC.broadcast(new int[]{dim, dim});

                        // [dc_1dW_ij dc_2dW_ij].transpose()
                        INDArray dC12 = Nd4j.concat(0, dXdXwi[start][split], dXdXwi[split][end]);

                        dXdXw[i][start][end][split] =
                                // f'(c1, c2) \dot
                                dC.muli(
                                // W *
                                model
                                        .getParams()
                                        .getW().mmul(
                                        // [dc_1 dc_2]^T))
                                        dC12));

                        // weighted marginalization over split
                        dXdXwi[start][end] = dXdXwi[start][end].add(
                                dXdXw[i][start][end][split].muli(
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
}
