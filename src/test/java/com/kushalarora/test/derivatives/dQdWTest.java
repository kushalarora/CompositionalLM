package com.kushalarora.test.derivatives;

import com.kushalarora.compositionalLM.derivatives.dQdW;
import com.kushalarora.compositionalLM.derivatives.dXdW;
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
public class dQdWTest extends AbstractDerivativeTest {
    private dQdW dqdw;

    @BeforeClass
    public static void setUpClass() {
        AbstractDerivativeTest.setUpClass();
        INDArray W = mock(INDArray.class);
        when(W.mmul((INDArray)any()))
                .thenReturn(Nd4j.ones(dim, 1));

        params.setW(W);

        INDArray  u = mock(INDArray.class);
        when(u.mmul((INDArray)any()))
                .thenReturn(Nd4j.ones(1));

        params.setU(u);
    }

    @Before
    public void setUp() {
        dqdw = new dQdW(dim, defaultSentence);
    }

    @Test
    public void testClear() {
        INDArray zeros = Nd4j.zeros(dim, 2 * dim);
        INDArray dW;
        dW = dqdw.calcDerivative(model, cScorer);

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
        INDArray dW = dqdw.calcDerivative(model, cScorer);

        INDArray truedW = Nd4j.zeros(dim, 2 * dim);

        INDArray ones = Nd4j.ones(dim, 2 * dim);
        for (int start = 0; start < length; start++) {
            for (int end = start + 1; end <= length; end++) {
                for  (int split = start + 1; split < end; split++)  {
                    truedW = truedW.add(ones);
                }
            }
        }
        double[][] compIScore = cScorer.getInsideSpanProb();
        truedW = truedW.div(compIScore[0][length]);

        assertEquals(dim*dim*2,
                truedW.eq(dW)
                        .sum(Integer.MAX_VALUE)
                        .getFloat(0,0), 1e-1);
    }
}
