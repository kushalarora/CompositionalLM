package com.kushalarora.compositionalLM.derivatives;

import com.kushalarora.compositionalLM.model.CompositionalInsideOutsideScore;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.optimizer.IIndexed;
import lombok.Getter;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

/**
 * Created by karora on 6/21/15.
 */
public class dQdW<T extends List<? extends IIndexed>> extends AbstractBaseDerivativeClass implements IDerivative<T> {
    @Getter
    private INDArray dQdW;
    private int dim;
    private T data;
    private int length;

    public dQdW(int dimension, T data) {
        super(new int[]{dimension, 2 * dimension});
        dim = dimension;
        this.dQdW = Nd4j.zeros(dim, 2 * dim);
        this.data = data;
        length = data.size();
    }

    public dQdW(dQdW dqdW, T data) {
        super(dqdW.dQdW.shape());
        dQdW = dqdW.dQdW.dup();
        dim = dqdW.dim;
        this.data = data;
        length = data.size();
    }

    private dQdW(INDArray dqdw, T data) {
        super(dqdw.shape());
        this.dQdW = dqdw;
        int[] shape = dqdw.shape();
        dim = shape[0];
        this.data = data;
        length = data.size();
    }

    public INDArray calcDerivative(Model model, CompositionalInsideOutsideScore scorer) {
        INDArray[][][][][] dxdwArr = new dXdW(dim, data).calcDerivative(model, scorer);
        INDArray[][][] compositionMatrix = scorer.getCompositionMatrix();
        double[][][] compositionalMu = scorer.getMuScore();
        double[][] compositionalIScore = scorer.getInsideSpanProb();
        INDArray[][] phraseMatrix = scorer.getPhraseMatrix();

        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < 2 * dim; j++) {
                double dEdW_ij = 0;

                for (int start = 0; start < length; start++) {
                    for (int end = start + 1; end <= length; end++) {
                        for (int split = start + 1; split < end; split++) {
                            double dE = model.energyDerivative(compositionMatrix[start][end][split],
                                    phraseMatrix[start][split], phraseMatrix[split][end]);
                            INDArray udXdWArr = model.getParams().getU().mmul(
                                    dxdwArr[i][j][start][end][split]);

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

                dQdW.putScalar(new int[]{i, j}, dEdW_ij);
            }
        }

        if (compositionalIScore[0][length] == 0) {
            throw new RuntimeException("Z is zero for sentence " + data);
        }

        dQdW = dQdW.div(compositionalIScore[0][length]);

        if (containsNanOrInf()) {
            return Nd4j.rand(dim, 2*dim, -1, 1, new JDKRandomGenerator());
        }
        return dQdW;
    }

    public void clear() {
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < 2 * dim; j++) {
                dQdW.putScalar(new int[]{i, j}, 0f);
            }
        }
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
                        ((dQdW) gradient).dQdW), data);
    }


}
