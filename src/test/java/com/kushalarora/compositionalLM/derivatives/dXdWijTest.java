package com.kushalarora.compositionalLM.derivatives;

import com.kushalarora.compositionalLM.options.Options;

import org.apache.commons.configuration.ConfigurationException;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;
import static org.mockito.Matchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Created by karora on 7/9/15.
 */
public class dXdWijTest extends AbstractDerivativeTest {

    protected dXdWij dxdw;

    @BeforeClass
    public static void setUpClass() throws ConfigurationException {
        AbstractDerivativeTest.setUpClass();
        INDArray W = mock(INDArray.class);
        when(W.mmul((INDArray) any()))
                .thenReturn(Nd4j.ones(dim, 1));

        params.setW(W);
    }

    @Before
    public void setUp() throws ConfigurationException {
        Options op = new Options();
        op.trainOp.modelParallel = true;
        dxdw = new dXdWij(dim, defaultSentence, op, 0, 0);
    }

    @Test
    public void testCalcDerivative() {
        INDArray[][] arr = dxdw.calcDerivative(model, cScorer);
        int i = 0, j = 0;

        for (int diff = 2; diff <= length; diff++) {
            for (int start = 0; start + diff <= length; start++) {
                int end = start + diff;
                INDArray dxdwij = Nd4j.zeros(dim, 1);
                for (int split = start + 1; split < end; split++) {
                    INDArray j1 = Nd4j.zeros(dim, 1);
                    j1.putScalar((j < dim ? j : j - dim), 1);
                    j1.addi(Nd4j.ones(dim, 1));
                    dxdwij.addi(j1);
                }
                double[][] compIScore = cScorer.getCompIScores();
                dxdwij.divi(compIScore[start][end]);
                assertEquals(dim,
                        dxdwij.eq(arr[start][end])
                                .sum(Integer.MAX_VALUE)
                                .getFloat(0),
                        1e-1);
            }
        }
    }

}
