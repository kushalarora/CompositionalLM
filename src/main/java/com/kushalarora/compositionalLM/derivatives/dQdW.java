package com.kushalarora.compositionalLM.derivatives;

import com.kushalarora.compositionalLM.model.CompositionalGrammar;
import com.kushalarora.compositionalLM.model.Model;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by karora on 6/21/15.
 */
public class dQdW extends AbstractBaseDerivativeClass implements IDerivative {
    private final dXdW dxdw;
    private INDArray dEdW;
    private int dim;

    public dQdW(Model model, dXdW dxdw) {
        super(model);
        this.dxdw = dxdw;
        dim = model.getDimensions();
        this.dEdW = Nd4j.zeros(dim, 2 * dim);
    }

    public INDArray calcDerivative(CompositionalGrammar.CompositionalInsideOutsideScorer scorer) {
        INDArray[][][][][] dxdwArr = dxdw.calcDerivative(scorer);
        int length = scorer.getCurrentSentence().size();
        INDArray[][][] compositionMatrix = scorer.getCompositionMatrix();
        float[][][] compositionalMu = scorer.getMuScore();
        INDArray[][] phraseMatrix = scorer.getPhraseMatrix();

        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < 2 * dim; j++) {
                float dEdW_ij = 0;

                for (int start = 0; start < length; start++) {
                    for (int end = start + 1; end <= length; end++) {
                        for (int split = start + 1; split < end; split++) {
                            float dE = model.energyDerivative(compositionMatrix[start][end][split],
                                    phraseMatrix[start][split], phraseMatrix[split][end]);
                            INDArray udXdWArr = model.getU().mmul(
                                    dxdwArr[i][j][start][end][split]);

                            int[] udXdWShape = udXdWArr.shape();
                            if (udXdWShape.length != 1 ||
                                    udXdWShape[0] != 1) {
                                throw new RuntimeException("udXdWArr was expected to be a matrix of shape 1 X 1");
                            }

                            float udXdW = udXdWArr.getFloat(0);
                            dEdW_ij += dE * udXdW * compositionalMu[start][end][split];
                        }
                    }
                }

                dEdW.putScalar(new int[]{i, j}, dEdW_ij);
            }
        }
        return dEdW;
    }

    public void clear() {
        dxdw.clear();
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < 2 * dim; j++) {
                dEdW.putScalar(new int[]{i, j}, 0f);
            }
        }
    }
}
