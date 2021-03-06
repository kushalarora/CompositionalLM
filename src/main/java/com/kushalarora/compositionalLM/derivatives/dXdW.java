package com.kushalarora.compositionalLM.derivatives;

import com.google.common.base.Function;
import com.kushalarora.compositionalLM.lang.StanfordCompositionalInsideOutsideScore;
import com.kushalarora.compositionalLM.model.CompositionalInsideOutsideScore;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.optimizer.IIndexed;
import com.kushalarora.compositionalLM.optimizer.IIndexedSized;
import com.kushalarora.compositionalLM.options.Options;
import com.kushalarora.compositionalLM.utils.Parallelizer;

import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

import javax.annotation.Nullable;

/**
 * Created by karora on 6/21/15.
 */
public class dXdW<T extends IIndexedSized> {
    // d X 2d array of column vectors
    @Getter
    private INDArray[][][][][] dXdW;
    private int dim;
    private int length;
    private T data;
    private Options op;
    private Parallelizer parallelizer;

    public dXdW(int dimension, T data, Options op) {
        dim = dimension;
        dXdW = new INDArray[dim][2 * dim][][][];
        this.data = data;
        length = data.getSize();
        this.op = op;
        parallelizer = new Parallelizer(op, op.grammarOp.maxLength/op.trainOp.blockNum + 1);

        // Allocate memory to hold spans and split
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < 2 * dim; j++) {
                dXdW[i][j] = new INDArray[length][length + 1][];
                for (int start = 0; start < length; start++) {
                    for (int end = start + 1; end <= length; end++) {
                        dXdW[i][j][start][end] = new INDArray[length];
                    }
                }
            }
        }
    }

    public INDArray[][][][][] calcDerivative(final Model model,
                                             final StanfordCompositionalInsideOutsideScore scorer) {

        final INDArray[][] phraseMatrix = scorer.getPhraseMatrix();
        final double[][][] compositionISplitScore = scorer.getCompISplitScore();
        final double[][] compositionIScore = scorer.getCompIScores();


        for (int i = 0; i < dim; i++) {
                final int iFinal = i;
                Function<Integer, Void> func = new Function<Integer, Void>()
                {
                    @Nullable
                    public Void apply(Integer j)
                    {
                        INDArray[][] dXdWij = new INDArray[length][length + 1];

                        // Swipe the span array clean for this i, j computation
                        for (int start = 0; start < length; start++) {
                            for (int end = start + 1; end <= length; end++) {
                                dXdWij[start][end] = Nd4j.zeros(dim, 1);
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
                                    INDArray vec = Nd4j.zeros(dim, 1);
                                    vec.putScalar((j < dim ? j : j - dim),
                                                  (j < dim ?
                                                          child1.getDouble(j) :
                                                          child2.getDouble(j - dim)));

                                    // [dc_1dW_ij dc_2dW_ij].transpose()
                                    INDArray dC12 = Nd4j.concat(0,
                                                                dXdWij[start][split],
                                                                dXdWij[split][end]);

                                    synchronized (dXdW) {
                                        dXdW[iFinal][j][start][end][split] =
                                                // (
                                                // 1_j \dot c_12 + (
                                                vec.add(
                                                        // W *
                                                        model
                                                                .getParams()
                                                                .getW().mmul(
                                                                // [dc_1 dc_2]^T)) *
                                                                dC12)).mul(
                                                        // \dot  f'(c_1, c_2)
                                                        dC);
                                    }

                                    // weighted marginalization over split
                                    // dXdW_ij += dXW_ij[k] * \pi(start,end,split)
                                    dXdWij[start][end] =
                                            dXdWij[start][end].add(
                                                    dXdW[iFinal][j][start][end][split].mul(
                                                            // \pi[start][end][split]
                                                            compositionISplitScore[start][end][split]));
                                }
                                if (compositionIScore[start][end] != 0){
                                    // dXdW_ij /= \pi(start,end)
                                    dXdWij[start][end] = dXdWij[start][end].div(
                                            compositionIScore[start][end]);
                                }
                            }
                        }
                        return null;
                    }
                };

            if (op.trainOp.parallel) {
                parallelizer.parallelizer(0, 2 * dim, func);
            } else
            {
                for (int j = 0; j < 2 * dim; j++)
                {
                    func.apply(j);
                }
            }
        }
        return dXdW;
    }
}
