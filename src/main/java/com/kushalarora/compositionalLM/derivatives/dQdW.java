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

@Slf4j
public class dQdW<T extends IIndexedSized> extends AbstractBaseDerivativeClass<T> implements IDerivative<T>
{
    @Getter
    private INDArray dQdW;
    private int dim;
    private int length;
    private Options op;
    private Parallelizer parallelizer;

    public dQdW(int dimension, T data, Options op) {
        super(new int[] {dimension, 2 * dimension}, data);
        dim = dimension;
        this.dQdW = Nd4j.zeros(dim, 2 * dim);
        length = data.getSize();
        this.op = op;
        parallelizer = new Parallelizer(op, op.grammarOp.maxLength / op.trainOp.blockNum + 1);
    }

    public dQdW(dQdW dqdW, T data, Options op)
    {
        super(dqdW.dQdW.shape(), data);
        dQdW = dqdW.dQdW.dup();
        dim = dqdW.dim;
        length = data.getSize();
        this.op = op;
        parallelizer = new Parallelizer(op, op.grammarOp.maxLength / op.trainOp.blockNum + 1);
    }

    private dQdW(INDArray dqdw, T data, Options op)
    {
        super(dqdw.shape(), data);
        this.dQdW = dqdw;
        int[] shape = dqdw.shape();
        dim = shape[0];
        length = data.getSize();
        this.op = op;
        parallelizer = new Parallelizer(op, op.grammarOp.maxLength / op.trainOp.blockNum + 1);
    }

    public static INDArray dEdWBinary(INDArray dXdWijParent,
                                      INDArray dXdWijChild1, INDArray dXdWijChild2,
                                      INDArray parent,
                                      INDArray child1, INDArray child2, Model model) {

	    double dE = model.energyCompDerivative(parent, child1, child2);

        INDArray dEdWBinary =
	        model.linearComposition(dXdWijParent,
		            dXdWijChild1,
		            dXdWijChild2)
		            .mul(dE);

	    int[] dEdWBinaryShape = dEdWBinary.shape();
	    if (dEdWBinaryShape[0] != 1 &&
		        dEdWBinaryShape[1] != 1) {
		    throw new RuntimeException("Dim should be 1X1.");
	    }

	    return dEdWBinary;
    }

    public void  calcDerivative(final Model model, final StanfordCompositionalInsideOutsideScore scorer) {

        if (length < 2) {
            // There is nothing to do here.
            return;
        }

        final INDArray[][][] compositionMatrix = scorer.getCompositionMatrix();
        final double[][][] compositionalMu = scorer.getCompMuScores();
        final double[][] compositionalIScore = scorer.getCompIScores();
        final INDArray[][] phraseMatrix = scorer.getPhraseMatrix();

        Function<Integer, Void> funci = new Function<Integer, Void>() {
            @Nullable
            public Void apply(final Integer i) {
                Function<Integer, Void> funcj = new Function<Integer, Void>() {
                    @Nullable
                    public Void apply(final Integer j) {

                        final INDArray[][] dxdwArr =
                            new dXdWij(dim, data, op, i, j)
                                .calcDerivative(model, scorer);

                        INDArray dEdW_ij = Nd4j.zeros(1,1);
                        INDArray[] dEW_ij_l = new INDArray[length];

                        for (int diff = 2; diff <= length; diff++) {
                            for (int st = 0; st + diff <= length; st++) {
                                final int start = st;
                                final int end = start + diff;

                                for (int split = start + 1; split < end; split++) {
                                    INDArray lineardXdW = dEdWBinary(
                                                            dxdwArr[start][end],
                                                            dxdwArr[start][split],
                                                            dxdwArr[split][end],
                                                            compositionMatrix[start][end][split],
                                                            phraseMatrix[start][split],
                                                            phraseMatrix[split][end],
                                                            model);

                                    dEW_ij_l[split] = lineardXdW;

                                    synchronized (dEdW_ij) {
                                        dEdW_ij = dEdW_ij
                                            .add(lineardXdW
                                                    .mul(compositionalMu[start][end][split]));
                                    }
                                }

                                double compMuSum = 0;
                                for (int sp = start + 1; sp < end; sp++) {
                                    compMuSum += compositionalMu[start][end][sp];
                                }

                                dEdW_ij = dEdW_ij.sub(model
                                            .Expectedl(
                                                start, end, dEW_ij_l,
                                                compositionMatrix[start][end],
                                                phraseMatrix,
                                                compMuSum, new int[]{1, 1}));
                            }
                        }

                        int[] dEdW_ijShape = dEdW_ij.shape();
                        if (dEdW_ijShape[0] != 1 && dEdW_ijShape[1] != 1) {
                            throw new RuntimeException("udXdWArr was expected to be a matrix of shape 1 X 1");
                        }
                        double dEdW_ijVal = dEdW_ij.getDouble(0, 0);
                        dQdW.putScalar(new int[]{i, j}, dEdW_ijVal);

                        return null;
                    }
                };

                if (op.trainOp.modelParallel) {
                    parallelizer.parallelizer(0, 2 * dim, funcj);
                } else {
                    for (int j = 0; j < 2 * dim; j++) {
                        funcj.apply(j);
                    }
                }
                return null;
            }
        };


        if (op.trainOp.modelParallel) {
            parallelizer.parallelizer(0, dim, funci);
        } else {
            for (int i = 0; i < dim; i++) {
                funci.apply(i);
            }
        }


        if (compositionalIScore[0][length] != 0) {
            double tmp = Math.pow(10, 6);
            dQdW = dQdW.div(compositionalIScore[0][length] * tmp).div(tmp);
        }


        if (containsNanOrInf()) {
            log.error("dQdW contains Nan Or Inf. for data {}::{}. Norm::{}", data.getIndex(), data.getSize(), norm());
            dQdW = Nd4j.zeros(dim, 2 * dim);
        }

        dQdW = clampDerivativeIfNeeded(dQdW);
    }

    public void clear() {
        dQdW = Nd4j.zeros(dim, 2 * dim);
    }

    public void add(IDerivative other) {
        dQdW = dQdW.add(((dQdW) other).getDQdW());
    }

    public void mul(double learningRate) {
        dQdW = dQdW.mul(learningRate);
    }

    public boolean containsNanOrInf() {
        return containsNanOrInf(dQdW);
    }

    public IDerivative adaGrad(IDerivative gradient) {
        return new dQdW(adaGrad.getGradient(
                        ((dQdW) gradient).dQdW, 100), data, op);
    }

    public double norm()
    {
        return Nd4j.norm2(dQdW).sum(Integer.MAX_VALUE).getDouble(0);
    }


}
