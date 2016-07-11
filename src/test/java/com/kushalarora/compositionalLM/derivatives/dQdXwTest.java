package com.kushalarora.compositionalLM.derivatives;

import com.google.common.base.Function;
import com.kushalarora.compositionalLM.lang.Word;
import com.kushalarora.compositionalLM.options.Options;

import org.apache.commons.configuration.ConfigurationException;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.Map;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertEquals;
import static org.mockito.Matchers.any;
import static org.mockito.Matchers.anyDouble;
import static org.mockito.Matchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Created by karora on 7/9/15.
 */
public class dQdXwTest extends AbstractDerivativeTest {

    private dQdXw dqdxw;

    @Before
    public void setUp() throws ConfigurationException
    {
        Options op = new Options();
        op.trainOp.modelParallel = true;
        dqdxw = new dQdXw(dim, V, defaultSentence, op);
    }

    @BeforeClass
    public static void setUpClass() throws ConfigurationException
    {
        AbstractDerivativeTest.setUpClass();
        INDArray W = mock(INDArray.class);
        when(W.mmul((INDArray) any()))
                .thenReturn(Nd4j.eye(dim));

        params.setW(W);

        INDArray  u = mock(INDArray.class);
        when(u.mmul((INDArray)any()))
                .thenReturn(Nd4j.ones(dim));

       params.setU(u);

        INDArray  h1 = mock(INDArray.class);
        when(h1.mmul((INDArray)any()))
            .thenReturn(Nd4j.ones(dim));

        params.setH1(h1);

        INDArray  h2 = mock(INDArray.class);
        when(h2.mmul((INDArray)any()))
            .thenReturn(Nd4j.ones(dim));

        params.setH2(h2);

        when(model.Expectedl(anyInt(),
            anyInt(), any(INDArray[].class),
            any(INDArray[].class),
                any(INDArray[][].class), anyDouble(),
                new int[]{dim, 1}))
            .thenReturn(Nd4j.zeros(dim, 1));

        when(model.ExpectedV(any(Function.class), any(int[].class)))
            .thenReturn(Nd4j.zeros(dim, 1));

        when(model.linearComposition(any(INDArray.class),
                any(INDArray.class), any(INDArray.class)))
            .thenReturn(Nd4j.ones(dim, 1));

        when(model.linearWord(any(INDArray.class)))
            .thenReturn(Nd4j.ones(dim, 1));
    }

    @Test
    public void  testClear() {

        dqdxw.calcDerivative(model, cScorer);

        Map<Integer, INDArray> dqdxwArr = dqdxw.getIndexToxMap();

        assertTrue(dqdxwArr.keySet().size() > 0);

        dqdxw.clear();

        dqdxwArr = dqdxw.getIndexToxMap();

        assertEquals(0, dqdxwArr.keySet().size());
    }

    @Test
    public void testCalcDerivative() {

        dqdxw.calcDerivative(model, cScorer);

        INDArray ones = Nd4j.ones(dim, 1);

        double[][] compIScore = cScorer.getCompIScores();

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
            Map<Integer, INDArray> indexToxMap = dqdxw.getIndexToxMap();
            List<Word> sentence = defaultSentence;
            assertEquals(dim ,
                    truedQdxwi.eq(indexToxMap.get(idx))
                            .sum(Integer.MAX_VALUE).getFloat(0),
                    1e-1);
        }
    }
}

