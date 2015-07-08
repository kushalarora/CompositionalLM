package com.kushalarora.test.lang;

import com.kushalarora.compositionalLM.lang.AbstractInsideOutsideScores;
import com.kushalarora.compositionalLM.lang.GrammarFactory;
import com.kushalarora.compositionalLM.lang.StanfordGrammar;
import com.kushalarora.compositionalLM.lang.Word;
import com.kushalarora.compositionalLM.options.Options;
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

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by karora on 6/25/15.
 */
@Slf4j
public class StanfordInsideOutsideScoresTest {

    private static int numStates;
    private static int length;
    private static List<Word> defaultSentence;
    private static StanfordGrammar sg;
    public AbstractInsideOutsideScores sIOScore;

    @BeforeClass
    public static void setUpClass() {
        val filePath = FileUtils.getFile("src/test/resources/wsjPCFG.ser.gz")
                .getAbsolutePath();

        sg = (StanfordGrammar) GrammarFactory.getGrammar(
                GrammarFactory.GrammarType.STANFORD_GRAMMAR,
                filePath, new Options());
        numStates = sg.getNumStates();
        PropertyConfigurator.configure("log4j.properties");
    }

    @Before
    public void setUp() {
        defaultSentence = new ArrayList<Word>();
        String[] sent = {"This", "is", "just", "a", "test", "."};
        for (String str : sent) {
            defaultSentence.add(new Word(str));
        }
        sIOScore = (StanfordGrammar.StanfordInsideOutsideScore)
                sg.getInsideOutsideObject(defaultSentence);
        length = sIOScore.getCurrentSentence().size();
    }

    @Test
    public void testIntializeArrays() {
        // All should be null at beginning

        //    assertEquals(length, sIOScore.getCurrentSentence().size());
        assertEquals(null, sIOScore.getInsideScores());
        assertEquals(null, sIOScore.getInsideSpanProb());
        assertEquals(null, sIOScore.getInsideSpanSplitProb());

        assertEquals(null, sIOScore.getOutsideScores());
        assertEquals(null, sIOScore.getOutsideSpanProb());
        assertEquals(null, sIOScore.getOutsideSpanWParentScore());

        assertEquals(null, sIOScore.getMuScore());
        assertEquals(null, sIOScore.getMuSpanSplitScore());

        sIOScore.initializeScoreArrays();


        float[][][] iScore = sIOScore.getInsideScores();
        float[][] iSpanScore = sIOScore.getInsideSpanProb();
        float[][][] iSpanSplitScore = sIOScore.getInsideSpanSplitProb();

        float[][][] oScore = sIOScore.getOutsideScores();
        float[][] oSpanScore = sIOScore.getOutsideSpanProb();
        float[][][] oSpanWParentScore = sIOScore.getOutsideSpanWParentScore();

        float[][][] muScore = sIOScore.getMuScore();
        float[][][] muSpanScore = sIOScore.getMuSpanSplitScore();

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
        float[] zerosNumStates = new float[numStates];
        float[] zerosLength = new float[length];
        float[] zerosLengthP1 = new float[length + 1];
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

        sIOScore.initializeScoreArrays();

        sIOScore.doLexScores();

        float[][] iSpanScores = sIOScore.getInsideSpanProb();
        float[][][] iScores = sIOScore.getInsideScores();

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

        sIOScore.initializeScoreArrays();
        sIOScore.doLexScores();

        sIOScore.doInsideScores();

        float[][] iSpanScores = sIOScore.getInsideSpanProb();
        float[][][] iScores = sIOScore.getInsideScores();
        float[][][] iSpanSplitScores = sIOScore.getInsideSpanSplitProb();

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

        sIOScore.initializeScoreArrays();
        sIOScore.doLexScores();
        sIOScore.doInsideScores();
        sIOScore.doOutsideScores();

        float[][] oSpanScores = sIOScore.getOutsideSpanProb();
        float[][][] oScores = sIOScore.getOutsideScores();
        float[][][] oSpanWParent = sIOScore.getOutsideSpanWParentScore();

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
        sIOScore.initializeScoreArrays();
        sIOScore.doLexScores();
        sIOScore.doInsideScores();
        sIOScore.doOutsideScores();
        sIOScore.computeMuSpanScore();

        float[][][] muSpanSplitScore = sIOScore.getMuSpanSplitScore();
        float[][][] muSpanStateScore = sIOScore.getMuScore();

        for (int start = 0; start < length; start++) {
            for (int end = start + 1; end <= length; end++) {
                float mu_score_state_sum = 0.0f;
                for (int state = 0; state < numStates; state++) {
                    mu_score_state_sum += muSpanStateScore[start][end][state];
                }

                float mu_score_split_sum = 0.0f;
                for (int split = start + 1; split < end; split++) {
                    mu_score_split_sum += muSpanSplitScore[start][end][split];
                }

                assertEquals(mu_score_split_sum, mu_score_state_sum);
            }
        }
    }
}
