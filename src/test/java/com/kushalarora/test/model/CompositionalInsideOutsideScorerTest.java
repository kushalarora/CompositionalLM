package com.kushalarora.test.model;

import com.kushalarora.compositionalLM.lang.GrammarFactory;
import com.kushalarora.compositionalLM.lang.IInsideOutsideScorer;
import com.kushalarora.compositionalLM.lang.stanford.StanfordGrammar;
import com.kushalarora.compositionalLM.lang.Word;
import com.kushalarora.compositionalLM.model.CompositionalGrammar;
import com.kushalarora.compositionalLM.model.Model;
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

import static com.kushalarora.compositionalLM.lang.GrammarFactory.getGrammar;
import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Created by karora on 7/2/15.
 */

// TODO:: Use mockito to mock  prescorer and write
// better and smarter tests
public class CompositionalInsideOutsideScorerTest {

    private static StanfordGrammar sg;
    private static List<Word> defaultSentence;
    private static int length;
    private static CompositionalGrammar cg;
    private static IInsideOutsideScorer preScorer;
    private CompositionalGrammar.CompositionalInsideOutsideScorer scorer;
    private static int dim;
    private static Model model;

    @BeforeClass
    public static void setUpClass() {
        PropertyConfigurator.configure("log4j.properties");
        val filePath =
                FileUtils.getFile("src/resources/wsjPCFG.ser.gz")
                .getAbsolutePath();
        Options op = new Options();
        op.grammarOp.grammarType = GrammarFactory.GrammarType.STANFORD_GRAMMAR;
        op.grammarOp.filename =  filePath;

        sg = (StanfordGrammar)getGrammar(op);


        defaultSentence = new ArrayList<Word>();
        String[] sent = {"This", "is", "just", "a", "test", "."};

        for (String str : sent) {
            int index = (int)Math.random() * (sg.getVocabSize() + 1);
            defaultSentence.add(new Word(str, index));
        }

        preScorer = sg.computeScore(defaultSentence);

        length = defaultSentence.size();

        model = new Model(10, sg);
        dim = model.getDimensions();

        cg = new CompositionalGrammar(model, op);

    }

    @Before
    public void setUp() {
        scorer = cg.getScorer();
    }

    @Test
    public void testInitializeMatrices() {


        double[][] iScore = scorer.getInsideSpanProb();
        double[][][] muScore = scorer.getMuScore();
        INDArray[][][] compMatrix = scorer.getCompositionMatrix();
        INDArray[][] phraseMatrix = scorer.getPhraseMatrix();
        double[][][] iSplitScore = scorer.getCompositionISplitScore();

        assertEquals(iScore, null);
        assertEquals(muScore, null);
        assertEquals(compMatrix, null);
        assertEquals(phraseMatrix, null);
        assertEquals(iSplitScore, null);

        scorer.considerCreatingMatrices(length);
        scorer.initializeMatrices(length);

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

        scorer.initializeMatrices(length);
        scorer.doInsideScore(defaultSentence, length, preScorer);
        scorer.doMuScore(length, preScorer);


        boolean alliScoresZeros = true;
        boolean allmuScoresZeros = true;

        iScore = scorer.getInsideSpanProb();
        muScore = scorer.getMuScore();
        compMatrix = scorer.getCompositionMatrix();
        phraseMatrix = scorer.getPhraseMatrix();
        iSplitScore = scorer.getCompositionISplitScore();

        // verify all initialize to zero
        double[] zerosLength = new double[length];
        double[] zerosLengthP1 = new double[length + 1];
        INDArray zerosIndArray = Nd4j.create(dim, 1);
        Arrays.fill(zerosLength, 0f);
        Arrays.fill(zerosLengthP1, 0f);
        for (int diff = 2; diff <= length; diff++) {
            for (int start = 0; start + diff <= length; start++) {
                int end = start + diff;
                allmuScoresZeros &= Arrays.equals(muScore[start][end], zerosLength);
                alliScoresZeros &= Arrays.equals(iSplitScore[start][end], zerosLength);

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
        assertFalse(alliScoresZeros);
        assertFalse(allmuScoresZeros);

        scorer.initializeMatrices(length);

        alliScoresZeros  = true;
        allmuScoresZeros = true;
        for (int start = 0; start < length; start++) {
            for (int end = start + 1; end < length; end++) {
                allmuScoresZeros &= Arrays.equals(muScore[start][end], zerosLength);
                alliScoresZeros &= Arrays.equals(iSplitScore[start][end], zerosLength);

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

        assertTrue(alliScoresZeros);
        assertTrue(allmuScoresZeros);

    }

    @Test
    public void testDoInsideScores() {
        scorer.considerCreatingMatrices(length);
        scorer.initializeMatrices(length);
        scorer.doInsideScore(defaultSentence,length, preScorer);

        // test iscore and isplitscore sanity
        double[][] iScore = scorer.getInsideSpanProb();
        double[][][] iSplitScore = scorer.getCompositionISplitScore();

        for (int start = 0; start < length; start++) {
            int end = start + 1;
            int split = start;
            assertEquals(iScore[start][end], iSplitScore[start][end][split]);
        }

        for (int start = 0; start < length; start++) {
            for (int end = start + 2; end <= length; end++) {
                double iSplitSum = 0;
                for (int split = start + 1; split < end; split++) {
                    iSplitSum += iSplitScore[start][end][split];
                }
                assertEquals(1, iScore[start][end]/ iSplitSum, 0.0001);
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
                compSumVec = compSumVec.div(iScore[start][end]);
                assertEquals(phraseMatrix[start][end], compSumVec);
            }
        }
    }

    @Test
    @Ignore
    public void testDoOutsideScores() {
        // TODO:: Do we even need to test this
        // as it is not being used at all.
        // Maybe sanity check? If so, then how?
    }

    @Test
    @Ignore
    public void testDoMuScores() {
        // TODO:: Figure out how to test  this
    }
}
