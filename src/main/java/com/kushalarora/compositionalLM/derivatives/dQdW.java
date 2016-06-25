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

    public void calcDerivative(final Model model, final StanfordCompositionalInsideOutsideScore scorer)
    {
        final INDArray[][][][][] dxdwArr = new dXdW(dim, data, op).calcDerivative(model, scorer);
        final INDArray[][][] compositionMatrix = scorer.getCompositionMatrix();
        final double[][][] compositionalMu = scorer.getCompMuScores();
        final double[][] compositionalIScore = scorer.getCompIScores();
        final INDArray[][] phraseMatrix = scorer.getPhraseMatrix();

        for (int i = 0; i < dim; i++)
        {
            final int iF = i;
            Function<Integer, Void> func = new Function<Integer, Void>()
            {
                @Nullable
                public Void apply(Integer j)
                {
                    double dEdW_ij = 0;

                    for (int start = 0; start < length; start++) {
                        for (int end = start + 1; end <= length; end++) {
                            for (int split = start + 1; split < end; split++) {
                                double dE = model.energyDerivative(
                                        compositionMatrix[start][end][split],
                                        phraseMatrix[start][split],
                                        phraseMatrix[split][end]);

                                INDArray udXdWArr = model.getParams().getU().transpose().mmul(
                                        dxdwArr[iF][j][start][end][split]);

                                int[] udXdWShape = udXdWArr.shape();
                                if (udXdWShape[0] != 1 && udXdWShape[1] != 1) {
                                    throw new RuntimeException("udXdWArr was expected to be a matrix of shape 1 X 1");
                                }

                                double udXdW = udXdWArr.getDouble(0);
                                dEdW_ij += dE * udXdW * compositionalMu[start][end][split];
                            }
                        }
                    }

                    dQdW.putScalar(new int[]{iF, j}, dEdW_ij);
                    return null;
                }
            };

            if (op.trainOp.modelParallel) {
                parallelizer.parallelizer(0, 2 * dim, func);
            } else {
                for (int j = 0; j < 2 * dim; j++) {
                    func.apply(j);
                }
            }
        }

        if (compositionalIScore[0][length] == 0) {
            throw new RuntimeException("Z is zero for sentence " + data);
        }

        dQdW = dQdW.div(compositionalIScore[0][length]);

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
