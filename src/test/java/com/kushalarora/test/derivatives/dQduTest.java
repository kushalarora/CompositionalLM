package com.kushalarora.test.derivatives;

import com.kushalarora.compositionalLM.derivatives.dQdu;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * Created by karora on 7/9/15.
 */
public class dQduTest extends AbstractDerivativeTest {

    protected dQdu dQdu;

    @Before
    public void setUp() {
        dQdu = new dQdu(dim, defaultSentence);
    }

    /**
     *  As we have  mocked dE calculation and
     *  compositionMu, compositionMatrix and phraseMatrix
     *  to return 1. The whole calculation is reduced to
     *  adding the compositions.
     *
     *  We test that they are same.
     */
    @Test
    public void testCalcDerivative() {
        INDArray arr = dQdu.calcDerivative(model, cScorer);

        INDArray trueArray = Nd4j.zeros(dim, 1);

        INDArray[][] phraseMatrix = cScorer.getPhraseMatrix();
        INDArray[][][] compMatrix = cScorer.getCompositionMatrix();

        for (int idx = 0; idx < length; idx++) {
            trueArray = trueArray.add(phraseMatrix[idx][idx + 1]);
        }


        for (int diff = 2; diff <= length; diff++) {
            for (int start = 0; start + diff < length; start++) {
                int end = start + diff;
                for (int split = start + 1; split < end; split++) {
                    trueArray = trueArray.add(compMatrix[start][end][split]);
                }
            }
        }

        double[][] compIScore = cScorer.getInsideSpanProb();
        trueArray = trueArray.div(compIScore[0][length]);

        assertEquals(dim,
                trueArray.eq(arr).sum(Integer.MAX_VALUE).getFloat(0),
                1e-1);
    }


    @Test
    public void testClean() {
        INDArray arr = dQdu.calcDerivative(model, cScorer);
        INDArray trueArray = Nd4j.zeros(dim, 1);

        // As we called calcDerivative, the arr shouldn't be all zeros
        assertEquals(dim,
                trueArray.neq(arr).sum(Integer.MAX_VALUE).getFloat(0),
                1e-1);

        dQdu.clear();

        arr = dQdu.getDQdu();
        // Clear has been called, hence arr should be all zeros
        assertEquals(dim,
                trueArray.eq(arr).sum(Integer.MAX_VALUE).getFloat(0),
                1e-1);
    }
}
