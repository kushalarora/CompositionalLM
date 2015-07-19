package com.kushalarora.compositionalLM.derivatives;

import com.kushalarora.compositionalLM.lang.Word;
import com.kushalarora.compositionalLM.model.CompositionalGrammar;
import com.kushalarora.compositionalLM.model.Model;
import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

/**
 * Created by karora on 6/21/15.
 */
public class dQdW extends AbstractBaseDerivativeClass implements IDerivative {
    private final dXdW dxdw;
    @Getter
    private INDArray dQdW;
    private int dim;

    public dQdW(Model model, dXdW dxdw) {
        super(model);
        this.dxdw = dxdw;
        dim = model.getDimensions();
        this.dQdW = Nd4j.zeros(dim, 2 * dim);
    }

    public dQdW(Model model) {
        this(model, new dXdW(model));
    }


    public dQdW(dQdW dqdW) {
        super(dqdW.model);
        dxdw = dqdW.dxdw;
        dQdW = dqdW.dQdW;
        dim = dqdW.dim;
    }

    public INDArray calcDerivative(List<Word> sentence, CompositionalGrammar.CompositionalInsideOutsideScorer scorer) {
        int length = sentence.size();
        INDArray[][][][][] dxdwArr = dxdw.calcDerivative(sentence, scorer);
        INDArray[][][] compositionMatrix = scorer.getCompositionMatrix();
        float[][][] compositionalMu = scorer.getMuScore();
        float[][] compositionalIScore = scorer.getInsideSpanProb();
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
        dQdW = dQdW.div(compositionalIScore[0][length]);
        return dQdW;
    }

    public void clear() {
        dxdw.clear();
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < 2 * dim; j++) {
                dQdW.putScalar(new int[]{i, j}, 0f);
            }
        }
    }

    public IDerivative add(IDerivative other) {
        dQdW = dQdW.add(((dQdW)other).getDQdW());
        return this;
    }

    public IDerivative mul(double learningRate) {
        dQdW = dQdW.mul(learningRate);
        return this;
    }
}
