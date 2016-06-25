package com.kushalarora.test.derivatives;

import com.kushalarora.compositionalLM.derivatives.dXdXw;
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
public class dXdXwTest extends AbstractDerivativeTest {
    private dXdXw dxdxw;
    private static Options op;

    @BeforeClass
    public static void setUpClass() throws ConfigurationException
    {
        AbstractDerivativeTest.setUpClass();
        INDArray W = mock(INDArray.class);
        when(W.mmul((INDArray) any()))
                .thenReturn(Nd4j.eye(dim));

        params.setW(W);

        op = new Options();
        op.trainOp.modelParallel = true;
    }

    @Before
    public void setUp() {
        dxdxw = new dXdXw(dim, V, defaultSentence, op);
    }


    @Test
    public void testCalcDerivative() {
        INDArray[][][][] dxdxwArr = dxdxw.calcDerivative(model, cScorer);

        INDArray eye = Nd4j.eye(dim);
        INDArray truedxdw = Nd4j.zeros(dim, dim);

        for (int idx = 0; idx < length; idx++) {

            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    INDArray truedxdwi = Nd4j.eye(dim);
                    assertEquals(dim * dim,
                            truedxdwi.eq(dxdxw.getDXdXw()[idx][start][end][start])
                                    .sum(Integer.MAX_VALUE).getFloat(0, 0), 1e-1);
                }
            }

            for (int diff = 2; diff <= length; diff++) {
                for (int start = 0; start + diff <= length; start++) {
                    int end = start + diff;
                    for (int split = start + 1; split < end; split++) {
                        truedxdw = eye;
                        assertEquals(dim * dim,
                                truedxdw.eq(dxdxw.getDXdXw()[idx][start][end][split])
                                        .sum(Integer.MAX_VALUE).getFloat(0, 0), 1e-1);
                    }
                }
            }
        }
    }
}
