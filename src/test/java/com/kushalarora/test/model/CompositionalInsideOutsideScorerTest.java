package com.kushalarora.test.model;

import com.kushalarora.compositionalLM.lang.*;
import com.kushalarora.compositionalLM.model.CompositionalGrammar;
import com.kushalarora.compositionalLM.model.CompositionalInsideOutsideScore;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.options.Options;
import lombok.val;
import org.apache.commons.configuration.ConfigurationException;
import org.apache.commons.io.FileUtils;
import org.apache.log4j.PropertyConfigurator;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

import static com.kushalarora.compositionalLM.lang.GrammarFactory.getGrammar;
import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertFalse;

/**
 * Created by karora on 7/2/15.
 */

// TODO:: Use mockito to mock  prescorer and write
// better and smarter tests
public class CompositionalInsideOutsideScorerTest {

    private static StanfordGrammar sg;
    private static Sentence defaultSentence;
    private static int length;
    private static CompositionalGrammar cg;
    private static IInsideOutsideScore preScorer;
    private CompositionalInsideOutsideScore score;
    private static int dim;
    private static Model model;

    @BeforeClass
    public static void setUpClass() throws ConfigurationException {
        PropertyConfigurator.configure("log4j.properties");
        val filePath =
                FileUtils.getFile("src/resources/wsjPCFG.ser.gz")
                .getAbsolutePath();
        Options op = new Options();
        op.grammarOp.grammarType = GrammarFactory.GrammarType.STANFORD_GRAMMAR;
        op.grammarOp.filename =  filePath;

        sg = (StanfordGrammar)getGrammar(op);


        defaultSentence = new Sentence(0);
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
        score = new CompositionalInsideOutsideScore(defaultSentence, model.getDimensions());
    }

    @Test
    public void testInitializeMatrices() {

        double[][] iScore = score.getInsideSpanProb();
        double[][][] muScore = score.getMuScore();
        INDArray[][][] compMatrix = score.getCompositionMatrix();
        INDArray[][] phraseMatrix = score.getPhraseMatrix();
        double[][][] iSplitScore = score.getCompositionISplitScore();

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

        cg.doInsideScore(score, preScorer);
        cg.doMuScore(score, preScorer);


        boolean alliScoresZeros = true;
        boolean allmuScoresZeros = true;

        muScore = score.getMuScore();
        compMatrix = score.getCompositionMatrix();
        phraseMatrix = score.getPhraseMatrix();
        iSplitScore = score.getCompositionISplitScore();

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
    }

    @Test
    public void testDoInsideScores() {

        cg.doInsideScore(score, preScorer);

        // test iscore and isplitscore sanity
        double[][] iScore = score.getInsideSpanProb();
        double[][][] iSplitScore = score.getCompositionISplitScore();

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
        INDArray[][][] compMatrix = score.getCompositionMatrix();
        INDArray[][] phraseMatrix = score.getPhraseMatrix();

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
