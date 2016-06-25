package com.kushalarora.compositionalLM.derivatives;

import javax.annotation.Nullable;

import com.kushalarora.compositionalLM.lang.StanfordCompositionalInsideOutsideScore;
import com.kushalarora.compositionalLM.optimizer.IIndexedSized;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.google.common.base.Function;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.options.Options;
import com.kushalarora.compositionalLM.utils.Parallelizer;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;

/**
 * Created by karora on 6/21/15.
 */

@Slf4j
public class dXdXw<T extends IIndexedSized> {
    @Getter
    private INDArray[][][][] dXdXw;
    private int dim;
    private int V;
    private T data;
    private int length;
    private Options op;
    private Parallelizer parallelizer;

    public dXdXw(int dimension, int vocab, T data, Options op) {
        dim = dimension;
        V = vocab;
        this.data = data;
        length = data.getSize();
        dXdXw = new INDArray[length][][][];

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
        this.op = op;
        parallelizer = new Parallelizer(op, op.grammarOp.maxLength/op.trainOp.blockNum + 1);
    }

    public INDArray[][][][] calcDerivative(final Model model,
                                           final StanfordCompositionalInsideOutsideScore scorer) {

        final INDArray[][] phraseMatrix = scorer.getPhraseMatrix();
        final double[][][] compositionISplitScore = scorer.getCompISplitScore();
        final double[][] compositionIScore = scorer.getCompIScores();

        Function<Integer, Void> func = new Function<Integer, Void>() {
            @Nullable
            public Void apply(Integer i) {
                INDArray[][] dXdXwi = new INDArray[length][length + 1];
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

                for (int diff = 2; diff <= length; diff++) {
                    for (int start = 0; start + diff <= length; start++) {
                        int end = start + diff;
                        for (int split = start + 1; split < end; split++) {

                            // Calculate f'(c_1, c_2)
                            INDArray child1 = phraseMatrix[start][split];
                            INDArray child2 = phraseMatrix[split][end];
                            INDArray dC = model.composeDerivative(child1, child2);
                           dC = dC.transpose().broadcast(new int[] {dim, dim}).transpose();

                            // [dc_1dW_ij dc_2dW_ij].transpose()
                            INDArray dC12 = Nd4j.concat(0, dXdXwi[start][split], dXdXwi[split][end]);

                            dXdXw[i][start][end][split] =
                                    // f'(c1, c2) \dot
                                    dC.mul(
                                            // W *
                                            model
                                                    .getParams()
                                                    .getW().mmul(
                                                    // [dc_1 dc_2]^T))
                                                    dC12));

                            // weighted marginalization over split
                            dXdXwi[start][end] = dXdXwi[start][end].add(
                                    dXdXw[i][start][end][split].mul(
                                            // \pi[start][end][split]
                                            compositionISplitScore[start][end][split]));
                        }
                        if (compositionIScore[start][end] != 0) {
                            // dXdXwi /= \pi[start][end]
                            dXdXwi[start][end] = dXdXwi[start][end].div(
                                    compositionIScore[start][end]);

                        }
                    }
                }
                return null;
            }
        };

        if (op.trainOp.modelParallel) {
            parallelizer.parallelizer(0, length, func);
        } else {
            for (int i = 0; i < length; i++) {
                func.apply(i);
            }
        }
        return dXdXw;
    }
}
