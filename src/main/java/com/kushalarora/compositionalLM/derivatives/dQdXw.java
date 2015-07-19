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
public class dQdXw extends AbstractBaseDerivativeClass implements IDerivative {
    private dXdXw dxdxw;
    @Getter
    private INDArray dQdXw;
    private int dim;
    private int V;

    public dQdXw(Model model, dXdXw dxdxw) {
        super(model);
        this.dxdxw = dxdxw;
        dim = model.getDimensions();
        V = model.getVocabSize();
        dQdXw = Nd4j.zeros(dim, V);
    }


    public dQdXw(dQdXw dqdxw) {
        super(dqdxw.model);
        dQdXw = dqdxw.getDQdXw();
        dxdxw = dqdxw.dxdxw;
        dim = dqdxw.dim;
        V = dqdxw.V;
    }

    public dQdXw(Model model) {
        this(model, new dXdXw(model));
    }

    public INDArray calcDerivative(List<Word> sentence, CompositionalGrammar.CompositionalInsideOutsideScorer scorer) {

        int length = sentence.size();

        // Save indexes
        int[] indexes = new int[length];
        for (int i = 0; i < length; i++) {
            indexes[i] = sentence.get(i).getIndex();
        }

        INDArray[][][][] dxdxwArr = dxdxw.calcDerivative(sentence, scorer);
        INDArray[][][] compositionMatrix = scorer.getCompositionMatrix();
        float[][][] compositionalMu = scorer.getMuScore();
        INDArray[][] phraseMatrix = scorer.getPhraseMatrix();
        float[][] compositionalIScore = scorer.getInsideSpanProb();


        INDArray dcdc = Nd4j.eye(dim);
        INDArray dQdXw_i = Nd4j.zeros(dim, 1);
        for (int i = 0; i < length; i++) {

            // wipe the array clean
            for (int d = 0; d < dim; d++) {
                dQdXw_i.putScalar(d, 0f);
            }

            // handle leaf node
            INDArray vector = phraseMatrix[i][i + 1];
            float dE = model.energyDerivative(vector);

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
            dQdXw.add(Nd4j.zeros(dim, V));
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

    public IDerivative add(IDerivative other) {
        dQdXw = dQdXw.add(((dQdXw) other).getDQdXw());
        return this;
    }

    public IDerivative mul(double learningRate) {
        dQdXw = dQdXw.mul(learningRate);
        return this;
    }
}
