package com.kushalarora.compositionalLM.derivatives;

import com.kushalarora.compositionalLM.model.CompositionalGrammar;
import com.kushalarora.compositionalLM.model.Model;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by karora on 6/21/15.
 */
public class dXdW extends AbstractBaseDerivativeClass {
    // d X 2d array of column vectors
    INDArray[][][][][] dXdW;

    public dXdW(Model model) {
        super(model);
        int dim = model.params.getDimensions();
        dXdW = new INDArray[dim][2 * dim][][][];

    }

    public INDArray[][][][][] calcDerivative(CompositionalGrammar.CompositionalInsideOutsideScorer scorer) {
        int length = scorer.getCurrentSentence().size();
        int dim = model.params.getDimensions();

        // Allocate memory to hold spans and split
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < 2 * dim; j++) {
                dXdW[i][j] = new INDArray[length][][];
                for (int start = 0; start < length; start++) {
                    dXdW[i][j][start] = new INDArray[length][];
                    for (int end = start + 1; end <= length; end++) {
                        dXdW[i][j][start][end] = new INDArray[length];
                    }
                }
            }
        }


        INDArray[][] phraseMatrix = scorer.getPhraseMatrix();
        float[][][] compositionISplitScore = scorer.getCompositionISplitScore();
        float[][] compositionIScore = scorer.getInsideSpanProb();

        INDArray[][] dXdWij = new INDArray[length][length + 1];

        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < 2 * dim; j++) {


                // Swipe the span array clean for this i, j computation
                for (int start = 0; start < length; start++) {
                    for (int end = start + 1; end <= length; end++) {
                        for (int d = 0; d < dim; d++) {
                            dXdWij[start][end].putScalar(d, 0);
                        }
                    }
                }

                // for leaf nodes the derivative would be zero
                // as no W term is involved as their is no composition
                // for non leaf nodes
                for (int diff = 2; diff <= length; diff++) {
                    for (int start = 0; start + diff <= length; start++) {
                        int end = start + diff;
                        for (int split = start + 1; split < end; split++) {

                            // Calculate f'(c_1, c_2)
                            INDArray child1 = phraseMatrix[start][split];
                            INDArray child2 = phraseMatrix[split][end];
                            INDArray dC = model.composeDerivative(child1, child2);

                            // 1_j \dot c_12
                            INDArray vec = Nd4j.zeros(dim);
                            vec.putScalar(j, (j < dim ?
                                    child1.getFloat(j) :
                                    child2.getFloat(j - dim)));

                            // [dc_1dW_ij dc_2dW_ij].transpose()
                            INDArray dC12 = Nd4j.concat(0, dXdWij[start][split], dXdWij[split][end]);

                            dXdW[i][j][start][end][split] =
                                    // f'(c_1, c_2) \dot (
                                    dC.muli(
                                            // 1_j \dot c_12 + (
                                            vec.add(
                                                    // W *
                                                    model.params.getW().mul(
                                                            // [dc_1 dc_2]^T)) *
                                                            dC12)));


                            // weighted marginalization over split
                            // dXdW_ij += dXW_ij[k] * \pi(start,end,split)
                            dXdWij[start][end] =
                                    dXdWij[start][end].add(
                                            dXdW[i][j][start][end][split].muli(
                                                    // \pi[start][end][split]
                                                    compositionISplitScore[start][end][split]));
                        }

                        // dXdW_ij /= \pi(start,end)
                        dXdWij[start][end] = dXdWij[start][end].divi(
                                compositionIScore[start][end]);
                    }
                }
            }
        }
        return dXdW;
    }

    public void clear() {
        int dim = model.params.getDimensions();
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < 2 * dim; j++) {
                for (int k = 0; k < dim; k++) {
                    dXdW[i][j] = null;
                }
            }
        }
    }
}
