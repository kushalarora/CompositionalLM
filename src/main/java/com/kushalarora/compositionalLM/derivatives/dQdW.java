package com.kushalarora.compositionalLM.derivatives;

import java.util.List;

import javax.annotation.Nullable;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.google.common.base.Function;
import com.kushalarora.compositionalLM.model.CompositionalInsideOutsideScore;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.optimizer.IIndexed;
import com.kushalarora.compositionalLM.options.Options;
import com.kushalarora.compositionalLM.utils.Parallelizer;

import lombok.Getter;

/**
 * Created by karora on 6/21/15.
 */
public class dQdW<T extends List<? extends IIndexed>> extends AbstractBaseDerivativeClass implements IDerivative<T>
{
    @Getter
    private INDArray dQdW;
    private int dim;
    private T data;
    private int length;
    private Options op;
    private Parallelizer parallelizer;

    public dQdW(int dimension, T data, Options op)
    {
        super(new int[] {dimension, 2 * dimension});
        dim = dimension;
        this.dQdW = Nd4j.zeros(dim, 2 * dim);
        this.data = data;
        length = data.size();
        this.op = op;
        parallelizer = new Parallelizer(op, op.grammarOp.maxLength / op.trainOp.blockNum + 1);
    }

    public dQdW(dQdW dqdW, T data, Options op)
    {
        super(dqdW.dQdW.shape());
        dQdW = dqdW.dQdW.dup();
        dim = dqdW.dim;
        this.data = data;
        length = data.size();
        this.op = op;
        parallelizer = new Parallelizer(op, op.grammarOp.maxLength / op.trainOp.blockNum + 1);
    }

    private dQdW(INDArray dqdw, T data, Options op)
    {
        super(dqdw.shape());
        this.dQdW = dqdw;
        int[] shape = dqdw.shape();
        dim = shape[0];
        this.data = data;
        length = data.size();
        this.op = op;
        parallelizer = new Parallelizer(op, op.grammarOp.maxLength / op.trainOp.blockNum + 1);
    }

    public void calcDerivative(final Model model, final CompositionalInsideOutsideScore scorer)
    {
        final INDArray[][][][][] dxdwArr = new dXdW(dim, data, op).calcDerivative(model, scorer);
        final INDArray[][][] compositionMatrix = scorer.getCompositionMatrix();
        final double[][][] compositionalMu = scorer.getMuScore();
        final double[][] compositionalIScore = scorer.getInsideSpanProb();
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
                                double dE = model.energyDerivative(compositionMatrix[start][end][split],
                                          phraseMatrix[start][split], phraseMatrix[split][end]);
                                INDArray udXdWArr = model.getParams().getU().mmul(
                                        dxdwArr[iF][j][start][end][split]);

                                int[] udXdWShape = udXdWArr.shape();
                                if (udXdWShape.length != 1 ||
                                        udXdWShape[0] != 1) {
                                    throw new RuntimeException("udXdWArr was expected to be a matrix of shape 1 X 1");
                                }

                                double udXdW = udXdWArr.getFloat(0);
                                dEdW_ij += dE * udXdW * compositionalMu[start][end][split];
                            }
                        }
                    }

                    dQdW.putScalar(new int[]{iF, j}, dEdW_ij);
                    return null;
                }
            };

            if (op.trainOp.parallel) {
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
            dQdW = Nd4j.rand(dim, 2 * dim, -1, 1, new JDKRandomGenerator());
            dQdW = dQdW.div(Nd4j.norm2(dQdW));
        }
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
                        ((dQdW) gradient).dQdW), data, op);
    }

    public double norm()
    {
        return Nd4j.norm2(dQdW).sum(Integer.MAX_VALUE).getFloat(0);
    }


}
