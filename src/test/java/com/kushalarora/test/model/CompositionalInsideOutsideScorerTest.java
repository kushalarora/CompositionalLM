package com.kushalarora.test.model;

import com.kushalarora.compositionalLM.lang.GrammarFactory;
import com.kushalarora.compositionalLM.lang.StanfordGrammar;
import com.kushalarora.compositionalLM.lang.Word;
import com.kushalarora.compositionalLM.model.CompositionalGrammar;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.model.Parameters;
import com.kushalarora.compositionalLM.options.Options;
import lombok.val;
import org.apache.commons.io.FileUtils;
import org.apache.log4j.PropertyConfigurator;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Created by karora on 7/2/15.
 */
public class CompositionalInsideOutsideScorerTest {

    private static StanfordGrammar sg;
    private static List<Word> defaultSentence;
    private static int length;
    private static CompositionalGrammar cg;
    private static int dim;
    private static Model model;
    private CompositionalGrammar.CompositionalInsideOutsideScorer scorer;

    @BeforeClass
    public static void setUpClass() {
        val filePath = FileUtils.getFile("src/test/resources/wsjPCFG.ser.gz")
                .getAbsolutePath();
        sg = (StanfordGrammar) GrammarFactory.getGrammar(
                GrammarFactory.GrammarType.STANFORD_GRAMMAR,
                filePath, new Options());
        PropertyConfigurator.configure("log4j.properties");

        model = new Model(new Parameters(10, sg.getVocabSize()));
        dim = model.params.getDimensions();


    }

    @Before
    public void setUp() {
        defaultSentence = new ArrayList<Word>();
        String[] sent = {"This", "is", "just", "a", "test", "."};
        for (String str : sent) {
            defaultSentence.add(new Word(str));
        }
        length = defaultSentence.size();


        cg = new CompositionalGrammar(sg, model, new Options());
        scorer = cg.getScorer(defaultSentence);
    }

    @Test
    public void testInitializeMatrices() {

        float[][] iScore = scorer.getInsideSpanProb();
        float[][][] muScore = scorer.getMuScore();
        INDArray[][][] compMatrix = scorer.getCompositionMatrix();
        INDArray[][] phraseMatrix = scorer.getPhraseMatrix();
        float[][][] iSplitScore = scorer.getCompositionISplitScore();

        assertEquals(iScore, null);
        assertEquals(muScore, null);
        assertEquals(compMatrix, null);
        assertEquals(phraseMatrix, null);
        assertEquals(iSplitScore, null);

        scorer.createMatrices(length);

        iScore = scorer.getInsideSpanProb();
        muScore = scorer.getMuScore();
        compMatrix = scorer.getCompositionMatrix();
        phraseMatrix = scorer.getPhraseMatrix();
        iSplitScore = scorer.getCompositionISplitScore();

        assertEquals(iScore.length, length);
        assertEquals(muScore.length, length);
        assertEquals(compMatrix.length, length);
        assertEquals(phraseMatrix.length, length);
        assertEquals(iSplitScore.length, length);

        for (int start = 0; start < length; start++) {
            assertEquals(iScore[start].length, length + 1);
            assertEquals(muScore[start].length, length + 1);
            assertEquals(compMatrix[start].length, length + 1);
            assertEquals(phraseMatrix[start].length, length + 1);
            assertEquals(iSplitScore[start].length, length + 1);

            for (int end = start + 1; end < length; end++) {
                assertEquals(muScore[start][end].length, length);
                assertEquals(compMatrix[start][end].length, length);
                assertEquals("Start: " + start + " End: " + end, iSplitScore[start][end].length, length);
            }
        }


        scorer.doInsideScore();
        scorer.doMuScore();


        // scorer.initializeMatrices(length);

        iScore = scorer.getInsideSpanProb();
        muScore = scorer.getMuScore();
        compMatrix = scorer.getCompositionMatrix();
        phraseMatrix = scorer.getPhraseMatrix();
        iSplitScore = scorer.getCompositionISplitScore();

        // verify all initialize to zero
        float[] zerosLength = new float[length];
        float[] zerosLengthP1 = new float[length + 1];
        INDArray zerosIndArray = Nd4j.create(dim, 1);
        Arrays.fill(zerosLength, 0f);
        Arrays.fill(zerosLengthP1, 0f);
        for (int start = 0; start < length; start++) {
            for (int end = start + 1; end < length; end++) {
                assertFalse("Start: " + start + " End: " + end,
                        Arrays.equals(muScore[start][end], zerosLength));
                assertFalse("Start: " + start + " End: " + end,
                        Arrays.equals(iSplitScore[start][end], zerosLength));

                assertEquals(dim,
                        phraseMatrix[start][end].neq(
                                zerosIndArray
                        ).sum(Integer.MAX_VALUE).getFloat(0),
                        1e-1);

                for (int split = start + 1; split < end; split++) {
                    assertEquals(dim,
                            compMatrix[start][end][split].neq(
                                    zerosIndArray
                            ).sum(Integer.MAX_VALUE).getFloat(0),
                            1e-1);
                }
            }
        }

        scorer.initializeMatrices(length);

        for (int start = 0; start < length; start++) {
            for (int end = start + 1; end < length; end++) {
                //               assertFalse(Arrays.equals(muScore[start][end], zerosLength));
                assertTrue("Start: " + start + "End: " + end,
                        Arrays.equals(iSplitScore[start][end], zerosLength));

                assertEquals(dim,
                        phraseMatrix[start][end].eq(
                                zerosIndArray
                        ).sum(Integer.MAX_VALUE).getFloat(0),
                        1e-1);

                for (int split = start + 1; split < end; split++) {
                    assertEquals(dim,
                            compMatrix[start][end][split].eq(
                                    zerosIndArray
                            ).sum(Integer.MAX_VALUE).getFloat(0),
                            1e-1);
                }
            }
        }
    }

    @Test
    public void testDoInsideScores() {
        scorer.createMatrices(length);
        scorer.initializeMatrices(length);
        scorer.doInsideScore();

        // test iscore and isplitscore sanity
        float[][] iScore = scorer.getInsideSpanProb();
        float[][][] iSplitScore = scorer.getCompositionISplitScore();

        for (int start = 0; start < length; start++) {
            int end = start + 1;
            int split = start;
            assertEquals(iScore[start][end], iSplitScore[start][end][split]);
        }

        for (int start = 0; start < length; start++) {
            for (int end = start + 2; end <= length; end++) {
                float iSplitSum = 0;
                for (int split = start + 1; split < end; split++) {
                    iSplitSum += iSplitScore[start][end][split];
                }
                assertEquals(iScore[start][end], iSplitSum);
            }
        }

        // test phrase matrix and compositionMatrix sanity
        INDArray[][][] compMatrix = scorer.getCompositionMatrix();
        INDArray[][] phraseMatrix = scorer.getPhraseMatrix();

        for (int start = 0; start < length; start++) {
            for (int end = start + 2; end <= length; end++) {
                INDArray compSumVec = Nd4j.zeros(dim, 1);
                for (int split = start + 1; split < end; split++) {
                    compSumVec = compSumVec.add(
                            compMatrix[start][end][split].mul(
                                    iSplitScore[start][end][split]
                            ));
                }
                assertEquals(phraseMatrix[start][end], compSumVec);
            }
        }
    }

    @Test
    @Ignore
    public void testDoOutsideScores() {

    }

    @Test
    @Ignore
    public void testDoMuScores() {

    }
}
