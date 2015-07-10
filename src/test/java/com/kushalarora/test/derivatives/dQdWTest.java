package com.kushalarora.test.derivatives;

import com.kushalarora.compositionalLM.derivatives.dQdW;
import com.kushalarora.compositionalLM.derivatives.dXdW;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * Created by karora on 7/9/15.
 */
public class dQdWTest extends AbstractDerivativeTest {


    private dQdW dqdw;
    @Before
    public void setUp() {
        dqdw = new dQdW(model, new dXdW(model));
    }

    @Test
    public void testClear() {
        INDArray zeros = Nd4j.zeros(dim, 2 * dim);
        INDArray dW;
        dW = dqdw.calcDerivative(cScorer);

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
        INDArray dW = dqdw.calcDerivative(cScorer);

        INDArray truedW = Nd4j.zeros(dim, 2 * dim);

        INDArray ones = Nd4j.ones(dim, 2 * dim);
        for (int start = 0; start < length; start++) {
            for (int end = start + 1; end <= length; end++) {
                for  (int split = start + 1; split < end; split++)  {
                    truedW = truedW.add(ones);
                }
            }
        }

        assertEquals(dim*dim*2,
                truedW.eq(dW)
                        .sum(Integer.MAX_VALUE)
                        .getFloat(0,0), 1e-1);
    }
}
