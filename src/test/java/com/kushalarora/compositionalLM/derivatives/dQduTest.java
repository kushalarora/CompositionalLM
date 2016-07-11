package com.kushalarora.compositionalLM.derivatives;

import com.google.common.base.Function;
import com.kushalarora.compositionalLM.options.Options;

import org.apache.commons.configuration.ConfigurationException;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;
import static org.mockito.Matchers.any;
import static org.mockito.Matchers.anyDouble;
import static org.mockito.Matchers.anyInt;
import static org.mockito.Mockito.when;

/**
 * Created by karora on 7/9/15.
 */
public class dQduTest extends AbstractDerivativeTest {

    protected dQdu dqdu;

    @Before
    public void setUp() throws ConfigurationException
    {
        Options op = new Options();
        op.trainOp.modelParallel = false;
        dqdu = new dQdu(dim, defaultSentence, op);


        when(model.Expectedl(anyInt(),
                anyInt(), any(INDArray[].class),
                any(INDArray[].class), any(INDArray[][].class)
                , anyDouble(), new int[]{dim, 1}))
                .thenReturn(Nd4j.zeros(dim, 1));

        when(model.ExpectedV(any(Function.class), any(int[].class)))
                .thenReturn(Nd4j.zeros(dim, 1));

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
        dqdu.calcDerivative(model, cScorer);

        INDArray trueArray = Nd4j.zeros(dim, 1);

        INDArray[][] phraseMatrix = cScorer.getPhraseMatrix();
        INDArray[][][] compMatrix = cScorer.getCompositionMatrix();

        for (int idx = 0; idx < length; idx++) {
           trueArray.addi(phraseMatrix[idx][idx + 1]);
        }


        for (int diff = 2; diff <= length; diff++) {
            for (int start = 0; start + diff <= length; start++) {
                int end = start + diff;
                for (int split = start + 1; split < end; split++) {
                    trueArray.addi(compMatrix[start][end][split]);
                }
            }
        }

        double[][] compIScore = cScorer.getCompIScores();
        trueArray = trueArray.div(compIScore[0][length]);

        assertEquals(dim,
                trueArray.eq(dqdu.getDQdu()).sum(Integer.MAX_VALUE).getFloat(0),
                1e-1);
    }

    @Test
    public void testClean() {
        dqdu.calcDerivative(model, cScorer);
        INDArray arr = dqdu.getDQdu();

        INDArray trueArray = Nd4j.zeros(dim, 1);

        // As we called calcDerivative, the arr shouldn't be all zeros
        assertEquals(dim,
                trueArray.neq(arr).sum(Integer.MAX_VALUE).getFloat(0),
                1e-1);

        dqdu.clear();

        arr = dqdu.getDQdu();
        // Clear has been called, hence arr should be all zeros
        assertEquals(dim,
                trueArray.eq(arr).sum(Integer.MAX_VALUE).getFloat(0),
                1e-1);
    }
}
