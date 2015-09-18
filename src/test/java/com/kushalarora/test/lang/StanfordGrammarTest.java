package com.kushalarora.test.lang;

import com.kushalarora.compositionalLM.lang.*;
import com.kushalarora.compositionalLM.options.Options;
import org.apache.commons.configuration.ConfigurationException;
import org.apache.commons.io.FileUtils;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.ujmp.core.SparseMatrix;

import java.util.List;

import static com.kushalarora.compositionalLM.lang.GrammarFactory.getGrammar;
import static junit.framework.TestCase.assertEquals;

/**
 * Created by karora on 6/25/15.
 */
public class StanfordGrammarTest {

    public static String GRAMMAR_RELATIVE_FILE_PATH = "src/resources/englishPCFG.ser.gz";
    public static StanfordGrammar sg;
    public static List<Word> defaultSentence;
    public AbstractInsideOutsideScore sIOScore;
    public int length;


    @BeforeClass
    public static void setUpClass() throws ConfigurationException {
        String absoluteFilePath = FileUtils
                .getFile(GRAMMAR_RELATIVE_FILE_PATH)
                .getAbsolutePath();
        Options op = new Options();
        op.grammarOp.grammarType = GrammarFactory.GrammarType.STANFORD_GRAMMAR;
        op.grammarOp.filename =  absoluteFilePath;
        sg = (StanfordGrammar)getGrammar(op);

        String[] sent = {"This", "is", "just", "a", "test", "."};
        for (String str : sent) {
            int index = (int)Math.random() * (sg.getVocabSize() + 1);
            defaultSentence.add(new Word(str, index));
        }
    }

    @Before
    public void setUp() {
        sIOScore = new StanfordInsideOutsideScore((Sentence)defaultSentence, sg.getNumStates());
        length = sIOScore.getCurrentSentence().size();
    }

    @Test
    /**
     * All we can test is sanity that all marginalization worked.
     * So we can test that iScore marginalized over states, iSpanSplitScore
     * marginalized for splits equals ISpanScores
     */
    public void testDoLexScore() {

        sg.doLexScores(sIOScore);

        SparseMatrix iSpanScores = sIOScore.getInsideSpanProb();
        SparseMatrix  iScores = sIOScore.getInsideScores();

        for (int start = 0; start < length; start++) {
            for (int end = start + 1; end <= length; end++) {
                float iScores_start_end = 0;
                for (int state = 0; state < sg.getNumStates(); state++) {
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


        sg.doLexScores(sIOScore);
        sg.doInsideScores(sIOScore);

        SparseMatrix iSpanScores = sIOScore.getInsideSpanProb();
        SparseMatrix iScores = sIOScore.getInsideScores();
        SparseMatrix iSpanSplitScores = sIOScore.getInsideSpanSplitProb();

        for (int start = 0; start < length; start++) {
            for (int end = start + 2; end < length; end++) {
                float iScores_start_end = 0;
                for (int state = 0; state < sg.getNumStates(); state++) {
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

        sg.doLexScores(sIOScore);
        sg.doInsideScores(sIOScore);
        sg.doOutsideScores(sIOScore);

        SparseMatrix oScores = sIOScore.getOutsideScores();
        SparseMatrix oSpanWParent = sIOScore.getOutsideSpanWParentScore();

        for (int diff = length; diff >= 1; diff--) {
            for (int start = 0; start + diff <= length; start++) {
                int end = start + diff;
                float oScores_start_end = 0;

                for (int state = 0; state < sg.getNumStates(); state++) {
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


        sg.doLexScores(sIOScore);
        sg.doInsideScores(sIOScore);
        sg.doOutsideScores(sIOScore);
        sg.doMuScore(sIOScore);

        SparseMatrix muSpanStateScore = sIOScore.getMuScore();
        SparseMatrix muSpanSplitWParent = sIOScore.getMuSpanSplitScoreWParent();
        for (int start = 0; start < length; start++) {
            int end = start + 1;
            int split = start;

            float mu_score_state_sum = 0.0f;
            for (int state = 0; state < sg.getNumStates(); state++) {
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
                for (int state = 0; state < sg.getNumStates(); state++) {
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
