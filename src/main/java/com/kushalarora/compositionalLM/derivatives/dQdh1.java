package com.kushalarora.compositionalLM.derivatives;

import com.google.common.base.Function;
import com.kushalarora.compositionalLM.lang.StanfordCompositionalInsideOutsideScore;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.optimizer.IIndexedSized;
import com.kushalarora.compositionalLM.options.Options;
import com.kushalarora.compositionalLM.utils.Parallelizer;
import javax.annotation.Nullable;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by karora on 6/21/15.
 * Energy function E is given as
 * E = g(u^Tp) where p is phrase vector..
 * dEdu = g'(u.t().dot(p))p
 * <p/>
 * dQdh1 = \sum{start}{end}{split} dEdu(start, end, split) * \mu(start, end, split)
 */
@Slf4j
public class dQdh1<T extends IIndexedSized> extends AbstractBaseDerivativeClass<T> implements IDerivative<T> {
    @Getter
    private INDArray dQdh1;
    private int dimensions;
    private int length;
    private Options options;
    private Parallelizer parallelizer;

    public dQdh1(int dim, T data, Options op) {
        super(new int[]{dim, 1}, data);
        dQdh1 = Nd4j.zeros(dim, 1);
        dimensions = dim;
        this.data = data;
        length = data.getSize();
        options = op;
        parallelizer = new Parallelizer(op, op.grammarOp.maxLength/op.trainOp.blockNum + 1);
    }

    public dQdh1(dQdh1 dqdu, T data, Options op) {
        super(dqdu.dQdh1.shape(), data);
        dQdh1 = dqdu.dQdh1.dup();
        dimensions = dqdu.dQdh1.shape()[0];
        length = data.getSize();
        options = op;
        parallelizer = new Parallelizer(op, op.grammarOp.maxLength/op.trainOp.blockNum + 1);
    }

    private dQdh1(INDArray dqdu, T data, Options op) {
        super(dqdu.shape(), data);
        this.dQdh1 = dqdu;
        dimensions = dqdu.shape()[0];
        length = data.getSize();
        options = op;
        parallelizer = new Parallelizer(op, op.grammarOp.maxLength/op.trainOp.blockNum + 1);
    }

    public void clear() {
        // Wipe clean
        for (int i = 0; i < dimensions; i++) {
            dQdh1.putScalar(i, 0);
        }
    }

    public void add(IDerivative other) {
        dQdh1.addi(((dQdh1) other).getDQdh1());
    }

    public void mul(double learningRate) {
        dQdh1.muli(learningRate);
    }

    public boolean containsNanOrInf() {
        return containsNanOrInf(dQdh1);
    }

    public IDerivative adaGrad(IDerivative gradient) {
        return new dQdh1(adaGrad.getGradient(((dQdh1) gradient).dQdh1, 100), data, options);
    }

    public static INDArray dEduBinary(INDArray parent, INDArray child1, INDArray child2, Model model) {
        // dE = g'(u.t().dot(c1))
        double dE = model.energyCompDerivative(parent, child1, child2);

        // dEdu = g'(s) X u^T.dot(c1)
        return child1.mul(dE);
    }

    public double norm()
    {
        return Nd4j.norm2(dQdh1).sum(Integer.MAX_VALUE).getDouble(0);
    }

    public void calcDerivative(final Model model, final StanfordCompositionalInsideOutsideScore scorer) {
        if (length < 2) {
            // Nothing to do here.
            return;
        }
        final INDArray[][][] compositionMatrix = scorer.getCompositionMatrix();
        final INDArray[][] phraseMatrix = scorer.getPhraseMatrix();
        final double[][][] compositionMu = scorer.getCompMuScores();
        final double[][] compositionalIScore = scorer.getCompIScores();

        for (int diff = 2; diff <= length; diff++) {
            final int diffFinal = diff;
            for (int st = 0; st + diff <= length; st++) {
                final int start = st;
                final int end = start + diffFinal;

                final INDArray[] dEdh1 = new INDArray[length];

                Function<Integer, Void> binaryFunc = new Function<Integer, Void>() {
                    @Nullable
                    public Void apply(final Integer split) {

                        // dEdu = dE * p = g'(u.t().dot(p)) * c1
                        INDArray dEdh1s =
                            dEduBinary(compositionMatrix[start][end][split],
                                        phraseMatrix[start][split],
                                        phraseMatrix[split][end],
                                        model);

                        dEdh1[split] = dEdh1s;
                        synchronized (dQdh1) {
                            // dQdh1 * p(w) += dEdu * \mu[start][end][split]
                            dQdh1.addi(dEdh1s.muli(compositionMu[start][end][split]));
                        }

                        return null;
                    }
                };

                if (options.trainOp.modelParallel) {
                    parallelizer.parallelizer(start + 1, end, binaryFunc);
                } else {
                    for (int split = start + 1; split < end; split++) {
                        binaryFunc.apply(split);
                    }
                }

                double compMuSum = 0;
                for (int sp = start + 1; sp < end; sp++) {
                    compMuSum += compositionMu[start][end][sp];
                }

                dQdh1.subi(model
                        .Expectedl(start, end, dEdh1,
                            compositionMatrix[start][end],
                            phraseMatrix, compMuSum, new int[] {dimensions, 1}));
            }
        }

        if (compositionalIScore[0][length] != 0) {
            // dQdh1 = dQdh1 * p(w)/p(w)
            dQdh1.divi(compositionalIScore[0][length]);
        }


        if (containsNanOrInf()) {
            log.error("dQdh1 contains Nan Or Inf. data {}::{}. Norm::{}",
                    data.getIndex(), data.getSize(), norm());
            dQdh1 = Nd4j.zeros(dimensions, 1);
        }

        dQdh1 = clampDerivativeIfNeeded(dQdh1);
    }
}
