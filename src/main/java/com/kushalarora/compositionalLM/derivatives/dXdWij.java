package com.kushalarora.compositionalLM.derivatives;

import com.google.common.base.Function;
import com.kushalarora.compositionalLM.lang.StanfordCompositionalInsideOutsideScore;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.optimizer.IIndexedSized;
import com.kushalarora.compositionalLM.options.Options;
import com.kushalarora.compositionalLM.utils.Parallelizer;
import javax.annotation.Nullable;
import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class dXdWij<T extends IIndexedSized> {
    // d X 2d array of column vectors
    @Getter
    private INDArray[][] dXdWij;
    private int dim;
    private int length;
    private T data;
    private Options op;
    private Parallelizer parallelizer;
    private int i;
    private int j;

    public dXdWij(int dimension, T data, Options op, int i, int j) {
        dim = dimension;
        length = data.getSize();
        this.data = data;
        this.op = op;
        this.i = i;
        this.j = j;
        dXdWij = new INDArray[length][length + 1];
        parallelizer = new Parallelizer(op, op.grammarOp.maxLength/op.trainOp.blockNum + 1);
    }

    public INDArray[][] calcDerivative(final Model model,
                                             final StanfordCompositionalInsideOutsideScore scorer) {

        final INDArray[][] phraseMatrix = scorer.getPhraseMatrix();
        final double[][][] compositionISplitScore = scorer.getCompISplitScore();
        final double[][] compositionIScore = scorer.getCompIScores();

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
            for (int st = 0; st + diff <= length; st++) {
                final int start = st;
                final int end = start + diff;

                if (compositionIScore[start][end] == 0) {
                    continue;
                }

                Function<Integer, Void> splitFunction = new Function<Integer, Void>() {
                    @Nullable
                    public Void apply(@Nullable Integer split) {
                        // Calculate f'(c_1, c_2)
                        INDArray child1 = phraseMatrix[start][split];
                        INDArray child2 = phraseMatrix[split][end];
                        INDArray dC = model.composeDerivative(child1, child2);

                        INDArray vec = Nd4j.zeros(dim, 1);

                        // 1_j \dot c_12
                        vec.putScalar((j < dim ? j : j - dim),
                                        (j < dim ?
                                            child1.getDouble(j) :
                                            child2.getDouble(j - dim)));

                        double splitNorm =
                            compositionISplitScore[start][end][split]/compositionIScore[start][end];

                        // [dc_1dW_ij dc_2dW_ij].transpose()
                        INDArray dC12 = Nd4j.concat(0, dXdWij[start][split], dXdWij[split][end]);

                        // (1_j \dot c_12
                        vec = vec
                            // + (W
                            .addi(model
                                .getParams()
                                .getW()
                                //* [dc_1 dc_2]^T)))
                                .mmul(dC12))
                            // \dot  f'(c_1, c_2)
                            .muli(dC)
                            .muli(splitNorm);

                        // weighted marginalization over split
                        // dXdW_ij += dXW_ij[k] * \pi(start,end,split)
                        synchronized (dXdWij[start][end]) {
                            dXdWij[start][end] =
                                dXdWij[start][end].addi(vec);
                        }
                        return null;
                    }
                };


                if (op.trainOp.modelParallel) {
                    parallelizer.parallelizer(start + 1, end, splitFunction);
                } else {
                    for (int split = start + 1; split < end; split++) {
                        splitFunction.apply(split);
                    }
                }

            }
        }
        return dXdWij;
    }
}
