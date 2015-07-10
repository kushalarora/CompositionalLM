package com.kushalarora.test.derivatives;

import com.kushalarora.compositionalLM.lang.Word;
import com.kushalarora.compositionalLM.model.CompositionalGrammar;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.model.Parameters;
import org.junit.BeforeClass;
import org.nd4j.linalg.api.activation.Activations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jblas.NDArray;

import java.util.ArrayList;
import java.util.List;

import static org.mockito.Matchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Created by karora on 7/9/15.
 */
public class AbstractDerivativeTest {

    protected static Model model;
    protected static Parameters params;
    protected static int dim = 10;
    protected static int V = 100;
    protected static CompositionalGrammar.CompositionalInsideOutsideScorer cScorer;
    protected static int length;
    @BeforeClass
    public static void setUpClass() {
        model = mock(Model.class);
        params = mock(Parameters.class);
        cScorer = mock(CompositionalGrammar.CompositionalInsideOutsideScorer.class);
        List<Word> defaultSentence = new ArrayList<Word>();
        for (String str : new String[]{"This", "is", "just", "a", "test", "."}) {
            defaultSentence.add(new Word(str));
        }
        length = defaultSentence.size();

        when(cScorer.getCurrentSentence()).thenReturn(defaultSentence);

        INDArray[][][] dummyCompMatrix = new INDArray[length][length + 1][];
        INDArray[][] phraseMatrix = new INDArray[length][length + 1];
        float[][][] compMu = new float[length][length + 1][];

        float [][][] compISplitScore = new float[length][length + 1][];
        float [][] compIScore = new float[length][length + 1];


        for (int start = 0; start < length; start++) {
            for (int end = 0; end <= length; end++) {
                dummyCompMatrix[start][end] = new INDArray[length];
                compMu[start][end] = new float[length];
                compISplitScore[start][end] = new float[length];

                phraseMatrix[start][end] = Nd4j.ones(dim, 1);
                for (int split = start + 1; split < end; split++) {
                    dummyCompMatrix[start][end][split] = Nd4j.ones(dim, 1);
                }
            }
        }

        for (int idx = 0; idx < length; idx++) {
            compMu[idx][idx + 1][idx] = 1.0f;
            compISplitScore[idx][idx + 1][idx] = 1.0f;
            compIScore[idx][idx + 1] = 1.0f;
        }

        for (int start = 0; start < length; start++) {
            for (int end = 0; end <= length; end++) {
                compIScore[start][end] = end - start - 1;
                for (int split = start + 1; split < end; split++) {
                    compMu[start][end][split] = 1.0f;
                    compISplitScore[start][end][split] = 1.0f;
                }
            }
        }
        // compositionMatrix and phraseMatrix are mocked to
        // matrices of vectors of all 1
        when(cScorer.getCompositionMatrix())
                .thenReturn(dummyCompMatrix);

        when(cScorer.getPhraseMatrix())
                .thenReturn(phraseMatrix);

        // mu score is mocked to be 1 for all spans and splits
        when(cScorer.getMuScore())
                .thenReturn(compMu);

        // iSplit score is mocked to all 1
        when(cScorer.getCompositionISplitScore())
                .thenReturn(compISplitScore);

        // inside score is mocked to keep the sanity
        // with iSplitScore
        when(cScorer.getInsideSpanProb())
                .thenReturn(compIScore);

        // When asked for energy derivative, mock it to 1.0f
        when(model.energyDerivative((INDArray) any()))
                .thenReturn(1.0f);

        when(model.energyDerivative(
                (INDArray)any(), (INDArray)any(), (INDArray)any()))
                .thenReturn(1.0f);

        when(params.getDimensions())
                .thenReturn(dim);

        when(params.getVocabSize())
                .thenReturn(V);

        INDArray W = mock(INDArray.class);
        when(W.mmul((INDArray)any()))
                .thenReturn(Nd4j.ones(dim, 1));

        when(params.getW())
                .thenReturn(W);

        INDArray  u = mock(INDArray.class);
        when(u.mmul((INDArray)any()))
                .thenReturn(Nd4j.ones(1, 1));

        when(params.getU())
                .thenReturn(u);

        when(model.getParams())
                .thenReturn(params);

        // return all ones as composition derivative
        when(model.composeDerivative((INDArray)any(), (INDArray)any()))
                .thenReturn(Nd4j.ones(dim, 1));
    }
}
