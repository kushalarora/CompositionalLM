package com.kushalarora.compositionalLM.derivatives;

import com.google.common.base.Function;
import com.kushalarora.compositionalLM.lang.StanfordCompositionalInsideOutsideScore;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.optimizer.IIndexedSized;
import com.kushalarora.compositionalLM.options.Options;
import com.kushalarora.compositionalLM.utils.Parallelizer;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.annotation.Nullable;

/**
 * Created by karora on 6/21/15.
 * Energy function E is given as
 * E = g(u^Tp) where p is phrase vector..
 * dEdu = g'(u.t().dot(p))p
 * <p/>
 * dQdu = \sum{start}{end}{split} dEdu(start, end, split) * \mu(start, end, split)
 */
@Slf4j
public class dQdu<T extends IIndexedSized> extends AbstractBaseDerivativeClass<T> implements IDerivative<T> {
    @Getter
    private INDArray dQdu;
    private int dimensions;
    private int length;
    private Options options;
    private Parallelizer parallelizer;
    private static INDArray zLeaf;


    public dQdu(int dim, T data, Options op) {
        super(new int[]{dim, 1}, data);
        dQdu = Nd4j.zeros(dim, 1);
        dimensions = dim;
        this.data = data;
        length = data.getSize();
        options = op;
        parallelizer = new Parallelizer(op, op.grammarOp.maxLength/op.trainOp.blockNum + 1);
    }

    public dQdu(dQdu dqdu, T data, Options op) {
        super(dqdu.dQdu.shape(), data);
        dQdu = dqdu.dQdu.dup();
        dimensions = dqdu.dQdu.shape()[0];
        length = data.getSize();
        options = op;
        parallelizer = new Parallelizer(op, op.grammarOp.maxLength/op.trainOp.blockNum + 1);
    }

    private dQdu(INDArray dqdu, T data, Options op) {
        super(dqdu.shape(), data);
        this.dQdu = dqdu;
        dimensions = dqdu.shape()[0];
        length = data.getSize();
        options = op;
        parallelizer = new Parallelizer(op, op.grammarOp.maxLength/op.trainOp.blockNum + 1);
    }

    public void clear() {
        // Wipe clean
        for (int i = 0; i < dimensions; i++) {
            dQdu.putScalar(i, 0);
        }
    }

    public void add(IDerivative other) {
        dQdu.addi(((dQdu) other).getDQdu());
    }

    public void mul(double learningRate) {
        dQdu.muli(learningRate);
    }

    public boolean containsNanOrInf() {
        return containsNanOrInf(dQdu);
    }

    public IDerivative adaGrad(IDerivative gradient) {
        return new dQdu(adaGrad.getGradient(((dQdu) gradient).dQdu, 100), data, options);
    }

    public double norm()
    {
        return Nd4j.norm2(dQdu).sum(Integer.MAX_VALUE).getDouble(0);
    }

    public static INDArray dEduUnary(INDArray word, Model model) {
        // dE = g'(u.t().dot(p))
        double dE = model.energyWordDerivative(word);

        // dEdu = g'(s) * p
        return word.mul(dE);
    }

    public static INDArray dEduBinary(INDArray parent, INDArray child1, INDArray child2, Model model) {
        // dE = g'(u.t().dot(p))
        double dE = model.energyCompDerivative(parent, child1, child2);

        // dEdu = g'(s) X u^T.dot(c1)
        return parent.mul(dE);
    }

    public void calcDerivative(final Model model, final StanfordCompositionalInsideOutsideScore scorer) {
        final INDArray[][][] compositionMatrix = scorer.getCompositionMatrix();
        final INDArray[][] phraseMatrix = scorer.getPhraseMatrix();
        final double[][][] compositionMu = scorer.getCompMuScores();
        final double[][] compositionalIScore = scorer.getCompIScores();



        Function<Integer, Void> unaryFunc = new Function<Integer, Void>()
        {
            public Void apply(Integer start) {
                int end = start + 1;
                int split = start;

                // For leaf nodes we consider the phrase
                // dEdu = g'(u.dot(pa))pa
                INDArray dEdu = dEduUnary(phraseMatrix[start][end], model);

                synchronized (dQdu) {
                    // dQdu * p(w) += dEdu * mu(start, end, split)
                    dQdu.addi(dEdu.subi(ZLeaf_dEdu(model, dimensions))
                                .muli(compositionMu[start][end][split]));
                }
                return null;
            }
        };

        if (options.trainOp.modelParallel) {
            parallelizer.parallelizer(0, length, unaryFunc);
        } else {
            // do leaf nodes
            for (int start = 0; start < length; start++) {
                unaryFunc.apply(start);
            }
        }


        for (int diff = 2; diff <= length; diff++) {
            final int diffFinal = diff;
            for (int st = 0; st + diff <= length; st++) {
                final int start = st;
                final int end = start + diffFinal;

                final INDArray[] dEdu = new INDArray[length];

                Function<Integer, Void> binaryFunc = new Function<Integer, Void>() {
                    @Nullable
                    public Void apply(final Integer split) {

                        INDArray dEdus =
                            dEduBinary(compositionMatrix[start][end][split],
                                        phraseMatrix[start][split],
                                        phraseMatrix[split][end],
                                        model);

                        dEdu[split] = dEdus;
                        synchronized (dQdu) {
                            // dQdu * p(w) += dEdu * \mu[start][end][split]
                            dQdu.addi(dEdus
                                    .muli(compositionMu[start][end][split]));
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

                dQdu.subi(model
                            .Expectedl(start, end, dEdu,
                                        compositionMatrix[start][end],
                                        phraseMatrix, compMuSum,
                                        new int[]{dimensions, 1}));
            }
        }

        if (compositionalIScore[0][length] != 0) {
            // dQdu = dQdu * p(w)/p(w)
            dQdu.divi(compositionalIScore[0][length]);
        }

	    if (containsNanOrInf()) {
            log.error("dQdu contains Nan Or Inf. data {}::{}. Norm::{}",
                    data.getIndex(), data.getSize(), norm());
            dQdu = Nd4j.zeros(dimensions, 1);
        }

        dQdu = clampDerivativeIfNeeded(dQdu);
    }

    private static INDArray ZLeaf_dEdu(final Model model, int dimensions) {
        if (zLeaf == null) {
            log.info("Calculating ZLeaf_dEdu");
            zLeaf = model.ExpectedV(new Function<INDArray, INDArray>() {
                @Nullable
                public INDArray apply(@Nullable INDArray indArray) {
                    return dEduUnary(indArray, model);
                }
            }, new int[]{dimensions, 1});
        }
        return zLeaf;
    }

    public static void cleanZLeaf() {
        log.info("Cleaning up ZLeaf_dEdu");
        zLeaf = null;
    }
}
