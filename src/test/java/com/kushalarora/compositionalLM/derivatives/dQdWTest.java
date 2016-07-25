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
import static org.mockito.Matchers.anyDouble;
import static org.mockito.Matchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Created by karora on 7/9/15.
 */
public class dQdWTest extends AbstractDerivativeTest {
    private dQdW dqdw;
    private static Options op;

    @BeforeClass
    public static void setUpClass() throws ConfigurationException
    {
        AbstractDerivativeTest.setUpClass();
        INDArray W = mock(INDArray.class);
        when(W.mmul((INDArray)any()))
                .thenReturn(Nd4j.ones(dim, 1));

        params.setW(W);

        INDArray  u = mock(INDArray.class);
        when(u.mmul((INDArray)any()))
                .thenReturn(Nd4j.ones(1));

        params.setU(u);


        INDArray  h1 = mock(INDArray.class);
        when(h1.mmul((INDArray)any()))
            .thenReturn(Nd4j.ones(1));

        params.setH1(h1);


        INDArray  h2 = mock(INDArray.class);
        when(h2.mmul((INDArray)any()))
            .thenReturn(Nd4j.ones(1));

        params.setH2(h2);


        when(model.linearComposition(any(INDArray.class),
            any(INDArray.class),
            any(INDArray.class))).thenReturn(Nd4j.ones(1, 1).muli(3));

        when(model.Expectedl(anyInt(),
            anyInt(), any(INDArray[].class),
            any(INDArray[].class), any(INDArray[][].class)
            , anyDouble(), new int[]{1, 1}))
            .thenReturn(Nd4j.zeros(1, 1));

        op = new Options();
        op.trainOp.modelParallel = false;
        op.trainOp.dataParallel = false;
    }

    @Before
    public void setUp() {
        dqdw = new dQdW(dim, defaultSentence,op);
    }

    @Test
    public void testClear() {
        INDArray zeros = Nd4j.zeros(dim, 2 * dim);

        dqdw.calcDerivative(model, cScorer);
        INDArray dW = dqdw.getDQdW();

        assertEquals(dim*dim*2,
                zeros.neq(dW)
                        .sum(Integer.MAX_VALUE)
                        .getFloat(0,0), 1e-1);

        dqdw.clear();

        assertEquals(dim * dim * 2,
                zeros.eq(dW)
                        .sum(Integer.MAX_VALUE)
                        .getFloat(0, 0), 1e-1);
    }

    @Test
    public void testCalcDerivative() {
        dqdw.calcDerivative(model, cScorer);

        INDArray truedW = Nd4j.zeros(dim, 2 * dim);

        INDArray ones = Nd4j.ones(dim, 2 * dim);
        for (int start = 0; start < length; start++) {
            for (int end = start + 1; end <= length; end++) {
                for  (int split = start + 1; split < end; split++)  {
                    truedW = truedW.add(ones);
                }
            }
        }

        truedW.muli(3);
        double[][] compIScore = cScorer.getCompIScores();
        truedW = truedW.div(compIScore[0][length]);

        assertEquals(dim*dim*2,
                truedW.eq(dqdw.getDQdW())
                        .sum(Integer.MAX_VALUE)
                        .getFloat(0,0), 1e-1);
    }
}