package com.kushalarora.test.derivatives;

import com.kushalarora.compositionalLM.lang.Sentence;
import com.kushalarora.compositionalLM.lang.StanfordCompositionalInsideOutsideScore;
import com.kushalarora.compositionalLM.lang.Word;
import com.kushalarora.compositionalLM.model.CompositionalInsideOutsideScore;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.model.Parameters;

import org.apache.commons.configuration.ConfigurationException;
import org.junit.BeforeClass;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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
    protected static int dim = 10;
    protected static int V = 100;
    protected static StanfordCompositionalInsideOutsideScore cScorer;
    protected static int length;
    protected static Parameters params;
    protected static Sentence defaultSentence;

    @BeforeClass
    public static void setUpClass() throws ConfigurationException
    {
        model = mock(Model.class);
        params = new Parameters(dim, V);
        cScorer = mock(StanfordCompositionalInsideOutsideScore.class);
        defaultSentence = new Sentence(0);
        int index = 0;
        for (String str : new String[]{"This", "is", "just", "a", "test", "."}) {
            defaultSentence.add(new Word(str, index++));
        }
        length = defaultSentence.size();

        INDArray[][][] dummyCompMatrix = new INDArray[length][length + 1][];
        INDArray[][] phraseMatrix = new INDArray[length][length + 1];
        double[][][] compMu = new double[length][length + 1][];

        double [][][] compISplitScore = new double[length][length + 1][];
        double [][] compIScore = new double[length][length + 1];


        for (int start = 0; start < length; start++) {
            for (int end = 0; end <= length; end++) {
                dummyCompMatrix[start][end] = new INDArray[length];
                compMu[start][end] = new double[length];
                compISplitScore[start][end] = new double[length];

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
        when(cScorer.getCompMuScores())
                .thenReturn(compMu);

        // iSplit score is mocked to all 1
        when(cScorer.getCompISplitScore())
                .thenReturn(compISplitScore);

        // inside score is mocked to keep the sanity
        // with iSplitScore
        when(cScorer.getCompIScores())
                .thenReturn(compIScore);

        // When asked for energy derivative, mock it to 1.0f
        when(model.energyDerivative((INDArray) any()))
                .thenReturn(1.0d);

        when(model.energyDerivative(
                (INDArray) any(), (INDArray) any(), (INDArray) any()))
                .thenReturn(1.0d);

        when(model.getDimensions())
                .thenReturn(dim);

        when(model.getVocabSize())
                .thenReturn(V);

        when(model.getParams())
                .thenReturn(params);

        // return all ones as composition derivative
        when(model.composeDerivative((INDArray) any(), (INDArray) any()))
                .thenReturn(Nd4j.ones(dim, 1));

    }
}
