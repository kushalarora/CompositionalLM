package com.kushalarora.test.derivatives;

import com.kushalarora.compositionalLM.derivatives.dQdXw;
import com.kushalarora.compositionalLM.derivatives.dXdXw;
import com.kushalarora.compositionalLM.lang.Word;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.mockito.Matchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Created by karora on 7/9/15.
 */
public class dQdXwTest extends AbstractDerivativeTest {

    private dQdXw dqdxw;

    @Before
    public void setUp() {
        dqdxw = new dQdXw(dim, V, defaultSentence);
    }

    @BeforeClass
    public static void setUpClass() {
        AbstractDerivativeTest.setUpClass();
        INDArray W = mock(INDArray.class);
        when(W.mmul((INDArray) any()))
                .thenReturn(Nd4j.eye(dim));

        params.setW(W);

        INDArray  u = mock(INDArray.class);
        when(u.mmul((INDArray)any()))
                .thenReturn(Nd4j.ones(dim));

       params.setU(u);
    }

    @Test
    public void  testClear() {

        dqdxw.calcDerivative(model, cScorer);

        INDArray dqdxwArr = dqdxw.getDQdXw();

        INDArray zeros = Nd4j.zeros(V, dim);

        assertEquals(dim * length, zeros.neq(dqdxwArr).sum(Integer.MAX_VALUE).getFloat(0, 0), 1e-1);

        dqdxw.clear();

        dqdxwArr = dqdxw.getDQdXw();

        assertEquals(dim * V, zeros.eq(dqdxwArr).sum(Integer.MAX_VALUE).getFloat(0, 0), 1e-1);
    }

    @Test
    public void testCalcDerivative() {

        INDArray dqdxwArr  = dqdxw.calcDerivative(model, cScorer);

        INDArray ones = Nd4j.ones(dim, 1);

        double[][] compIScore = cScorer.getInsideSpanProb();

        for (int idx = 0; idx < length; idx++) {
            INDArray truedQdxwi = Nd4j.ones(dim, 1);
            for (int diff = 2; diff <= length; diff++) {
                for (int start = 0; start + diff <= length; start++) {
                    int end = start + diff;
                    for (int split = start + 1; split < end; split++) {
                        truedQdxwi = truedQdxwi.add(ones);
                    }
                }
            }

            truedQdxwi = truedQdxwi.div(compIScore[0][length]);

            List<Word> sentence = defaultSentence;
            assertEquals(dim ,
                    truedQdxwi.eq(
                            dqdxwArr
                                    .getColumn(
                                            sentence.get(idx).getIndex()))
                            .sum(Integer.MAX_VALUE).getFloat(0),
                    1e-1);
        }

    }
}

