package com.kushalarora.test.lang;

import com.kushalarora.compositionalLM.lang.AbstractInsideOutsideScore;
import com.kushalarora.compositionalLM.lang.GrammarFactory;
import com.kushalarora.compositionalLM.lang.Sentence;
import com.kushalarora.compositionalLM.lang.Word;
import com.kushalarora.compositionalLM.lang.stanford.StanfordGrammar;
import com.kushalarora.compositionalLM.options.Options;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.configuration.ConfigurationException;
import org.apache.commons.io.FileUtils;
import org.apache.log4j.PropertyConfigurator;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.ujmp.core.SparseMatrix;

import java.util.Arrays;

import static com.kushalarora.compositionalLM.lang.GrammarFactory.getGrammar;
import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by karora on 6/25/15.
 */


@Slf4j
public class StanfordInsideOutsideScoresTest {

    public AbstractInsideOutsideScore sIOScore;
    private int length;

    private static int numStates;
    private static Sentence defaultSentence;
    private static StanfordGrammar sg;


    @BeforeClass
    public static void setUpClass() throws ConfigurationException {
        val filePath = FileUtils.getFile("src/resources/wsjPCFG.ser.gz")
                .getAbsolutePath();

        Options op = new Options();
        op.grammarOp.grammarType = GrammarFactory.GrammarType.STANFORD_GRAMMAR;
        op.grammarOp.filename = filePath;
        sg = (StanfordGrammar) getGrammar(op);
        numStates = sg.getNumStates();
        PropertyConfigurator.configure("log4j.properties");

        defaultSentence = new Sentence(0);
        String[] sent = {"This", "is", "just", "a", "test", "."};
        for (String str : sent) {
            int index = (int) Math.random() * (sg.getVocabSize() + 1);
            defaultSentence.add(new Word(str, index));
        }
    }

    @Before
    public void setUp() {
        sIOScore = sg.getScore(defaultSentence);
        length = sIOScore.getCurrentSentence().size();
    }

   /* @Test
    public void testIntializeArrays() {
        // All should be null at beginning
        sIOScore.clearArrays();

        assertEquals(null, sIOScore.getInsideScores());
        assertEquals(null, sIOScore.getInsideSpanProb());
        assertEquals(null, sIOScore.getInsideSpanSplitProb());

        assertEquals(null, sIOScore.getOutsideScores());
        assertEquals(null, sIOScore.getOutsideSpanWParentScore());

        assertEquals(null, sIOScore.getMuScore());

        sIOScore.considerCreatingArrays();
        sIOScore.initializeScoreArrays();


        SparseMatrix iScore = sIOScore.getInsideScores();
        SparseMatrix iSpanScore = sIOScore.getInsideSpanProb();
        SparseMatrix iSpanSplitScore = sIOScore.getInsideSpanSplitProb();

        SparseMatrix oScore = sIOScore.getOutsideScores();
        SparseMatrix oSpanWParentScore = sIOScore.getOutsideSpanWParentScore();

        SparseMatrix muScore = sIOScore.getMuScore();

        // verify sizes
        for (int start = 0; start < length; start++) {
            assertEquals(iSpanScore[start].length, length + 1);
            for (int end = start + 1; end <= length; end++) {
                assertEquals(iScore[start][end].length, numStates);
                assertEquals(iSpanSplitScore[start][end].length, length);

                assertEquals(oScore[start][end].length, numStates);
                assertEquals(oSpanWParentScore[start][end].length, length + 1);

                assertEquals(muScore[start][end].length, numStates);
            }
        }

        // verify all initialize to zero
        double[] zerosNumStates = new double[numStates];
        double[] zerosLength = new double[length];
        double[] zerosLengthP1 = new double[length + 1];
        Arrays.fill(zerosNumStates, 0f);
        Arrays.fill(zerosLength, 0f);
        Arrays.fill(zerosLengthP1, 0f);
        for (int start = 0; start < length; start++) {
            assertTrue(Arrays.equals(iSpanScore[start], zerosLengthP1));
            for (int end = start + 1; end <= length; end++) {
                assertTrue(Arrays.equals(iScore[start][end], zerosNumStates));
                assertTrue(Arrays.equals(iSpanSplitScore[start][end], zerosLength));

                assertTrue(Arrays.equals(oScore[start][end], zerosNumStates));
                assertTrue(Arrays.equals(oSpanWParentScore[start][end], zerosLengthP1));

                assertTrue(Arrays.equals(muScore[start][end], zerosNumStates));
            }
        }
    }
*/
    @Test
    /**
     * All we can test is sanity that all marginalization worked.
     * So we can test that iScore marginalized over states, iSpanSplitScore
     * marginalized for splits equals ISpanScores
     */
    public void testDoLexScore() {

        sIOScore.clearArrays();
        sIOScore.considerCreatingArrays();
        sIOScore.initializeScoreArrays();
        sIOScore.doLexScores();

        SparseMatrix  iSpanScores = sIOScore.getInsideSpanProb();
        SparseMatrix  iScores = sIOScore.getInsideScores();

        for (int start = 0; start < length; start++) {
            for (int end = start + 1; end <= length; end++) {
                float iScores_start_end = 0;
                for (int state = 0; state < numStates; state++) {
                    iScores_start_end += sIOScore.getScore(iSpanScores, start, end, state);
                }
                assertEquals(sIOScore.getScore(iSpanScores, start, end), iScores_start_end, 0.0001);
            }
        }
    }

    @Test
    /**
     * All we can test is sanity that all marginalization worked.
     * So we can test that iScore marginalized over states, iSpanSplitScore
     * marginalized for splits equals ISpanScores
     */
    public void testDoInsideScore() {

        sIOScore.clearArrays();
        sIOScore.considerCreatingArrays();
        sIOScore.initializeScoreArrays();
        sIOScore.doLexScores();
        sIOScore.doInsideScores();

        SparseMatrix iSpanScores = sIOScore.getInsideSpanProb();
        SparseMatrix iScores = sIOScore.getInsideScores();
        SparseMatrix iSpanSplitScores = sIOScore.getInsideSpanSplitProb();

        for (int start = 0; start < length; start++) {
            for (int end = start + 2; end < length; end++) {
                float iScores_start_end = 0;
                for (int state = 0; state < numStates; state++) {
                    iScores_start_end += sIOScore.getScore(iScores, start, end, state);
                }
                float iScores_span_start_end = 0;
                for (int split = start + 1; split < length; split++) {
                    iScores_span_start_end += sIOScore.getScore(
                            iSpanSplitScores, start, end, split);
                }
                assertEquals("Start: " + start + " end: " +
                        end, iScores_start_end, iScores_span_start_end, .000001);
                assertEquals("Start: " + start + " end: " +
                        end, sIOScore.getScore(iSpanScores, start, end), iScores_start_end, .000001);
                assertEquals("Start: " + start + "end: " +
                        end, sIOScore.getScore(iSpanScores, start, end), iScores_span_start_end, .000001);

            }
        }
    }

    @Test
    /**
     * All we can test is sanity that all marginalization worked.
     * So we can test that iScore marginalized over states, iSpanSplitScore
     * marginalized for splits equals ISpanScores
     */
    public void testDoOutsideScore() {

        sIOScore.clearArrays();
        sIOScore.considerCreatingArrays();
        sIOScore.initializeScoreArrays();
        sIOScore.doLexScores();
        sIOScore.doInsideScores();
        sIOScore.doOutsideScores();

        SparseMatrix oScores = sIOScore.getOutsideScores();
        SparseMatrix oSpanWParent = sIOScore.getOutsideSpanWParentScore();

        for (int diff = length; diff >= 1; diff--) {
            for (int start = 0; start + diff <= length; start++) {
                int end = start + diff;
                float oScores_start_end = 0;

                for (int state = 0; state < numStates; state++) {
                    oScores_start_end += sIOScore.getScore(oScores, start, end, state);
                }
                float iScores_span_start_end = 0;
                for (int parent = 0; parent <= length; parent++) {
                    iScores_span_start_end += sIOScore.getScore(oSpanWParent, start, end, parent);
                }
            }
        }
    }

    @Test
    /**
     * Verify mu split score and mu state score are the same.
     */
    public void testDoMuScore() {

        sIOScore.clearArrays();
        sIOScore.considerCreatingArrays();
        sIOScore.initializeScoreArrays();
        sIOScore.doLexScores();
        sIOScore.doInsideScores();
        sIOScore.doOutsideScores();
        sIOScore.doMuScore();

        SparseMatrix muSpanStateScore = sIOScore.getMuScore();
        SparseMatrix muSpanSplitWParent = sIOScore.getMuSpanSplitScoreWParent();
        for (int start = 0; start < length; start++) {
            int end = start + 1;
            int split = start;

            float mu_score_state_sum = 0.0f;
            for (int state = 0; state < numStates; state++) {
                mu_score_state_sum += sIOScore.getScore(muSpanStateScore, start, end, state);
            }

            float mu_score_split_sum = 0.0f;
            float mu_score_split_parent_sum = 0;
            for (int parentL = 0; parentL < start; parentL++) {
                mu_score_split_parent_sum += sIOScore.getScore(
                        muSpanSplitWParent, start, end, split, parentL);
            }

            for (int parentR = 0; parentR < start; parentR++) {
                mu_score_split_parent_sum += sIOScore.getScore(
                        muSpanSplitWParent, start, end, split, parentR);
            }

            assertEquals("Start: " + start + " End: " + end,
                    mu_score_split_sum, mu_score_state_sum, 0.00001);

            assertEquals("Start: " + start + " End: " + end,
                    mu_score_split_sum, mu_score_split_parent_sum, 0.00001);
        }

        for (int diff = 2; diff <= length; diff++) {
            for (int start = 0; start + diff <= length; start++) {
                int end = start + diff;

                float mu_score_state_sum = 0.0f;
                for (int state = 0; state < numStates; state++) {
                    mu_score_state_sum += sIOScore.getScore(muSpanStateScore, start, end, state);
                }

                float mu_score_split_sum = 0.0f;
                float mu_score_split_parent_sum = 0;
                for (int split = start + 1; split < end; split++) {
                    for (int parentL = 0; parentL < start; parentL++) {
                        mu_score_split_parent_sum += sIOScore.getScore(
                                muSpanSplitWParent, start, end, split, parentL);
                    }

                    for (int parentR = 0; parentR < start; parentR++) {
                        mu_score_split_parent_sum += sIOScore.getScore(
                                muSpanSplitWParent, start, end, split, parentR);
                    }
                }

                assertEquals("Start: " + start + " End: " + end,
                        mu_score_split_sum, mu_score_state_sum, 0.00001);

                assertEquals("Start: " + start + " End: " + end,
                        mu_score_split_sum, mu_score_split_parent_sum, 0.00001);
            }
        }
    }
}
