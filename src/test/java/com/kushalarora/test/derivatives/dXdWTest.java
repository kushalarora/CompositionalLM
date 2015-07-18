package com.kushalarora.test.derivatives;

import com.kushalarora.compositionalLM.derivatives.dXdW;
import com.kushalarora.compositionalLM.model.Parameters;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static junit.framework.Assert.assertNull;
import static junit.framework.TestCase.assertNotNull;
import static org.junit.Assert.assertEquals;
import static org.mockito.Matchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Created by karora on 7/9/15.
 */
public class dXdWTest extends AbstractDerivativeTest {

    protected dXdW dxdw;

    @BeforeClass
    public static void setUpClass() {
        AbstractDerivativeTest.setUpClass();
        INDArray W = mock(INDArray.class);
        when(W.mmul((INDArray)any()))
                .thenReturn(Nd4j.ones(dim, 1));

        params.setW(W);
    }

    @Before
    public void setUp() {
        dxdw = new dXdW(model);
    }

    @Test
    public void testClear() {

        INDArray[][][][][] dxdwArr = dxdw.calcDerivative(defaultSentence, cScorer);
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < 2 * dim; j++) {
                assertNotNull(dxdwArr);
            }
        }

        dxdw.clear();

        dxdwArr = dxdw.getDXdW();
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < 2 * dim; j++) {
                assertNull(dxdwArr[i][j]);
            }
        }



    }

    @Test
    public void testCalcDerivative() {
        INDArray[][][][][] arr = dxdw.calcDerivative(defaultSentence, cScorer);
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < 2 *dim; j++) {
                INDArray j1 = Nd4j.zeros(dim, 1);
                j1.putScalar((j < dim ? j : j - dim), 1);

                for (int diff = 2; diff <= length; diff++) {
                    for (int start = 0; start + diff <= length; start++) {
                        int end = start + diff;
                        for (int split = start + 1; split < end; split++) {
                            INDArray vec = j1.add(Nd4j.ones(dim, 1));
                            assertEquals(dim,
                                    vec.eq(arr[i][j][start][end][split])
                                            .sum(Integer.MAX_VALUE).getFloat(0), 1e-1);
                        }
                    }
                }
            }
        }
    }

}
