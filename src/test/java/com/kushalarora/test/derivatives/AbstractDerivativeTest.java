package com.kushalarora.test.derivatives;

import com.kushalarora.compositionalLM.lang.Word;
import com.kushalarora.compositionalLM.model.CompositionalGrammar;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.model.Parameters;
import org.junit.BeforeClass;
import org.nd4j.linalg.api.activation.Activations;
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
    protected static CompositionalGrammar.CompositionalInsideOutsideScorer cScorer;
    protected static int length;
    @BeforeClass
    public static void setUpClass() {
        model = mock(Model.class);
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


        for (int start = 0; start < length; start++) {
            for (int end = 0; end <= length; end++) {
                dummyCompMatrix[start][end] = new INDArray[length];
                compMu[start][end] = new float[length];

                phraseMatrix[start][end] = Nd4j.ones(dim, 1);
                for (int split = start + 1; split < end; split++) {
                    dummyCompMatrix[start][end][split] = Nd4j.ones(dim, 1);
                }
            }
        }

        for (int idx = 0; idx < length; idx++) {
            compMu[idx][idx + 1][idx] = 1.0f;
        }

        for (int start = 0; start < length; start++) {
            for (int end = 0; end <= length; end++) {
                for (int split = start + 1; split < end; split++) {
                    compMu[start][end][split] = 1;
                }
            }
        }

        when(cScorer.getCompositionMatrix())
                .thenReturn(dummyCompMatrix);
        when(cScorer.getPhraseMatrix())
                .thenReturn(phraseMatrix);
        when(cScorer.getMuScore())
                .thenReturn(compMu);

        when(model.energyDerivative((INDArray) any()))
                .thenReturn(1.0f);
        when(model.getParams())
                .thenReturn(
                        new Parameters(10, 100,
                                Activations.linear(),
                                Activations.linear()));
    }
}
