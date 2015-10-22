package com.kushalarora.compositionalLM.derivatives;

import com.kushalarora.compositionalLM.model.CompositionalInsideOutsideScore;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.optimizer.IIndexed;
import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

/**
 * Created by karora on 6/21/15.
 */
public class dQdXw<T extends List<? extends IIndexed>> extends AbstractBaseDerivativeClass implements IDerivative<T> {
    @Getter
    private INDArray dQdXw;
    private int dim;
    private int V;
    private T data;
    private int length;

    public dQdXw(int dimensions, int vocabSize, T data) {
        super(new int[]{dimensions, dimensions});
        dim = dimensions;
        V = vocabSize;
        dQdXw = Nd4j.zeros(dim, V);
        this.data = data;
        length = data.size();
    }


    public dQdXw(dQdXw dqdxw, T data) {
        super(dqdxw.dQdXw.shape());
        dQdXw = dqdxw.dQdXw.dup();
        dim = dqdxw.dim;
        V = dqdxw.V;
        this.data = data;
        length = data.size();
    }

    private dQdXw(INDArray dqdxw, T data) {
        super(dqdxw.shape());
        dQdXw = dqdxw;
        int[] shape = dqdxw.shape();
        dim = shape[0];
        V = shape[1];
        this.data = data;
        length = data.size();
    }

    public INDArray calcDerivative(Model model, CompositionalInsideOutsideScore scorer) {

        // Save indexes
        int[] indexes = new int[length];
        for (int i = 0; i < length; i++) {
            indexes[i] = data.get(i).getIndex();
        }

        INDArray[][][][] dxdxwArr = new dXdXw(dim, V, data).calcDerivative(model, scorer);
        INDArray[][][] compositionMatrix = scorer.getCompositionMatrix();
        double[][][] compositionalMu = scorer.getMuScore();
        INDArray[][] phraseMatrix = scorer.getPhraseMatrix();
        double[][] compositionalIScore = scorer.getInsideSpanProb();


        INDArray dcdc = Nd4j.eye(dim);
        INDArray dQdXw_i = Nd4j.zeros(dim, 1);
        for (int i = 0; i < length; i++) {

            // wipe the array clean
            for (int d = 0; d < dim; d++) {
                dQdXw_i.putScalar(d, 0f);
            }

            // handle leaf node
            INDArray vector = phraseMatrix[i][i + 1];
            double dE = model.energyDerivative(vector);

            // diff wrt to self returns eye
            INDArray udXdXwArr =
                    model.getParams().getU()
                            .mmul(dcdc);

            int[] udXdXwShape = udXdXwArr.shape();
            if (udXdXwShape.length != 1 ||
                    udXdXwShape[0] != dim) {
                throw new RuntimeException("udXdXwArr was expected to be a matrix of shape dim X 1 " + udXdXwShape.toString());
            }

            dQdXw_i = dQdXw_i.add(udXdXwArr
                    .muli(compositionalMu[i][i + 1][i]))
                    .muli(dE);

            // handle the composition case
            for (int diff = 2; diff <= length; diff++) {
                for (int start = 0; start + diff <= length; start++) {
                    int end = start + diff;
                    for (int split = start + 1; split < end; split++) {
                        dE = model.energyDerivative(compositionMatrix[start][end][split],
                                phraseMatrix[start][split], phraseMatrix[split][end]);

                        udXdXwArr =
                                model
                                        .getParams()
                                        .getU()
                                        .mmul(dxdxwArr[i][start][end][split]);


                        udXdXwShape = udXdXwArr.shape();
                        if (udXdXwShape.length != 1 ||
                                udXdXwShape[0] != dim) {
                            throw new RuntimeException("udXdXwArr was expected to be a matrix of shape dim X 1");
                        }

                        dQdXw_i = dQdXw_i.add(udXdXwArr
                                .muli(compositionalMu[start][end][split]))
                                .muli(dE);
                    }
                }
            }
            int index = indexes[i];
            for (int d = 0; d < dim; d++) {
                dQdXw.putScalar(new int[]{d, index}, dQdXw_i.getFloat(d));
            }
        }

        if (compositionalIScore[0][length] == 0) {
            throw new RuntimeException("Z is zero for sentence " + data);
        }

        dQdXw = dQdXw.div(compositionalIScore[0][length]);
        return dQdXw;
    }

    public void clear() {
        // wipe the array
        for (int d = 0; d < dim; d++) {
            for (int v = 0; v < V; v++) {
                dQdXw.putScalar(new int[]{d, v}, 0f);
            }
        }

    }

    public void add(IDerivative other) {
        dQdXw = dQdXw.add(((dQdXw) other).getDQdXw());
    }

    public void mul(double learningRate) {
        dQdXw = dQdXw.mul(learningRate);
    }

    public boolean containsNanOrInf() {
        return containsNanOrInf(dQdXw);
    }

    public IDerivative adaGrad(IDerivative gradient) {
        return new dQdXw(adaGrad.getGradient(((dQdXw) gradient).dQdXw), data);
    }

}
