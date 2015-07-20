package com.kushalarora.test.lang;

import com.kushalarora.compositionalLM.lang.AbstractInsideOutsideScorer;
import com.kushalarora.compositionalLM.lang.GrammarFactory;
import com.kushalarora.compositionalLM.lang.stanford.StanfordGrammar;
import com.kushalarora.compositionalLM.lang.Word;
import com.kushalarora.compositionalLM.options.Options;
import edu.stanford.nlp.parser.lexparser.Lexicon;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.io.FileUtils;
import org.apache.log4j.PropertyConfigurator;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static com.kushalarora.compositionalLM.lang.GrammarFactory.getGrammar;
import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by karora on 6/25/15.
 */


@Slf4j
public class StanfordInsideOutsideScoresTest {

    public AbstractInsideOutsideScorer sIOScore;
    private int length;

    private static int numStates;
    private static List<Word> defaultSentence;
    private static StanfordGrammar sg;


    @BeforeClass
    public static void setUpClass() {
        val filePath = FileUtils.getFile("src/resources/wsjPCFG.ser.gz")
                .getAbsolutePath();

        Options op = new Options();
        op.grammarOp.grammarType = GrammarFactory.GrammarType.STANFORD_GRAMMAR;
        op.grammarOp.filename = filePath;
        sg = (StanfordGrammar) getGrammar(op);
        numStates = sg.getNumStates();
        PropertyConfigurator.configure("log4j.properties");

        defaultSentence = new ArrayList<Word>();
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

    @Test
    public void testIntializeArrays() {
        // All should be null at beginning
        sIOScore.clearArrays();

        assertEquals(null, sIOScore.getInsideScores());
        assertEquals(null, sIOScore.getInsideSpanProb());
        assertEquals(null, sIOScore.getInsideSpanSplitProb());

        assertEquals(null, sIOScore.getOutsideScores());
        assertEquals(null, sIOScore.getOutsideSpanProb());
        assertEquals(null, sIOScore.getOutsideSpanWParentScore());

        assertEquals(null, sIOScore.getMuScore());
        assertEquals(null, sIOScore.getMuSpanSplitScore());

        sIOScore.considerCreatingArrays();
        sIOScore.initializeScoreArrays();


        double[][][] iScore = sIOScore.getInsideScores();
        double[][] iSpanScore = sIOScore.getInsideSpanProb();
        double[][][] iSpanSplitScore = sIOScore.getInsideSpanSplitProb();

        double[][][] oScore = sIOScore.getOutsideScores();
        double[][] oSpanScore = sIOScore.getOutsideSpanProb();
        double[][][] oSpanWParentScore = sIOScore.getOutsideSpanWParentScore();

        double[][][] muScore = sIOScore.getMuScore();
        double[][][] muSpanScore = sIOScore.getMuSpanSplitScore();

        // verify sizes
        for (int start = 0; start < length; start++) {
            assertEquals(iSpanScore[start].length, length + 1);
            assertEquals(oSpanScore[start].length, length + 1);
            for (int end = start + 1; end <= length; end++) {
                assertEquals(iScore[start][end].length, numStates);
                assertEquals(iSpanSplitScore[start][end].length, length);

                assertEquals(oScore[start][end].length, numStates);
                assertEquals(oSpanWParentScore[start][end].length, length + 1);

                assertEquals(muScore[start][end].length, numStates);
                assertEquals(muSpanScore[start][end].length, length);
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
            assertTrue(Arrays.equals(oSpanScore[start], zerosLengthP1));
            for (int end = start + 1; end <= length; end++) {
                assertTrue(Arrays.equals(iScore[start][end], zerosNumStates));
                assertTrue(Arrays.equals(iSpanSplitScore[start][end], zerosLength));

                assertTrue(Arrays.equals(oScore[start][end], zerosNumStates));
                assertTrue(Arrays.equals(oSpanWParentScore[start][end], zerosLengthP1));

                assertTrue(Arrays.equals(muScore[start][end], zerosNumStates));
                assertTrue(Arrays.equals(muSpanScore[start][end], zerosLength));
            }
        }
    }

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

        double[][] iSpanScores = sIOScore.getInsideSpanProb();
        double[][][] iScores = sIOScore.getInsideScores();

        for (int start = 0; start < length; start++) {
            for (int end = start + 1; end <= length; end++) {
                float iScores_start_end = 0;
                for (int state = 0; state < numStates; state++) {
                    iScores_start_end += iScores[start][end][state];
                }
                assertEquals(iSpanScores[start][end], iScores_start_end, 0.0001);
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

        double[][] iSpanScores = sIOScore.getInsideSpanProb();
        double[][][] iScores = sIOScore.getInsideScores();
        double[][][] iSpanSplitScores = sIOScore.getInsideSpanSplitProb();

        for (int start = 0; start < length; start++) {
            for (int end = start + 2; end < length; end++) {
                float iScores_start_end = 0;
                for (int state = 0; state < numStates; state++) {
                    iScores_start_end += iScores[start][end][state];
                }
                float iScores_span_start_end = 0;
                for (int split = start + 1; split < length; split++) {
                    iScores_span_start_end += iSpanSplitScores[start][end][split];
                }
                assertEquals("Start: " + start + " end: " +
                        end, iScores_start_end, iScores_span_start_end, .000001);
                assertEquals("Start: " + start + " end: " +
                        end, iSpanScores[start][end], iScores_start_end, .000001);
                assertEquals("Start: " + start + "end: " +
                        end, iSpanScores[start][end], iScores_span_start_end, .000001);

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

        double[][] oSpanScores = sIOScore.getOutsideSpanProb();
        double[][][] oScores = sIOScore.getOutsideScores();
        double[][][] oSpanWParent = sIOScore.getOutsideSpanWParentScore();

        for (int diff = length; diff >= 1; diff--) {
            for (int start = 0; start + diff <= length; start++) {
                int end = start + diff;
                float oScores_start_end = 0;

                for (int state = 0; state < numStates; state++) {
                    oScores_start_end += oScores[start][end][state];
                }
                float iScores_span_start_end = 0;
                for (int parent = 0; parent <= length; parent++) {
                    iScores_span_start_end += oSpanWParent[start][end][parent];
                }

                assertEquals("Start:" + start + " End:" + end, oSpanScores[start][end], oScores_start_end, .0001);
                assertEquals("Start:" + start + " End:" + end, oSpanScores[start][end], iScores_span_start_end, .0001);
                log.debug("oSpanScores[{}][{}]={}", start, end, oSpanScores[start][end]);
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

        double[][][] muSpanSplitScore = sIOScore.getMuSpanSplitScore();
        double[][][] muSpanStateScore = sIOScore.getMuScore();
        double[][][][] muSpanSplitWParent = sIOScore.getMuSpanScoreWParent();
        for (int start = 0; start < length; start++) {
            int end = start + 1;
            int split = start;

            float mu_score_state_sum = 0.0f;
            for (int state = 0; state < numStates; state++) {
                mu_score_state_sum += muSpanStateScore[start][end][state];
            }

            float mu_score_split_sum = 0.0f;
            float mu_score_split_parent_sum = 0;
            mu_score_split_sum += muSpanSplitScore[start][end][split];
            for (int parentL = 0; parentL < start; parentL++) {
                mu_score_split_parent_sum += muSpanSplitWParent[start][end][split][parentL];
            }

            for (int parentR = 0; parentR < start; parentR++) {
                mu_score_split_parent_sum += muSpanSplitWParent[start][end][split][parentR];
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
                    mu_score_state_sum += muSpanStateScore[start][end][state];
                }

                float mu_score_split_sum = 0.0f;
                float mu_score_split_parent_sum = 0;
                for (int split = start + 1; split < end; split++) {
                    mu_score_split_sum += muSpanSplitScore[start][end][split];
                    for (int parentL = 0; parentL < start; parentL++) {
                        mu_score_split_parent_sum += muSpanSplitWParent[start][end][split][parentL];
                    }

                    for (int parentR = 0; parentR < start; parentR++) {
                        mu_score_split_parent_sum += muSpanSplitWParent[start][end][split][parentR];
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
