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
public class dXdXwiTest extends AbstractDerivativeTest {
	private dXdXwi dxdxw;
	private static Options op;

	@BeforeClass
	public static void setUpClass() throws ConfigurationException {
		AbstractDerivativeTest.setUpClass();
		INDArray W = mock(INDArray.class);
		when(W.mmul((INDArray) any()))
			.thenReturn(Nd4j.eye(dim));

		params.setW(W);

		op = new Options();
		op.trainOp.modelParallel = false;
		op.trainOp.dataParallel = false;
	}

	@Before
	public void setUp() {
		dxdxw = new dXdXwi(dim, V, defaultSentence, op, 0);
	}


	@Test
	public void testCalcDerivative() {
		INDArray[][] dxdxwArr = dxdxw.calcDerivative(model, cScorer);

		INDArray eye = Nd4j.eye(dim);
		INDArray truedxdwi = Nd4j.eye(dim);
		assertEquals(dim * dim,
			truedxdwi.eq(dxdxwArr[0][1])
				.sum(Integer.MAX_VALUE).getFloat(0, 0), 1e-1);


		for (int start = 1; start < length; start++) {
			int end = start + 1;
			INDArray truedxdw = Nd4j.zeros(dim, dim);
			assertEquals(dim * dim,
				truedxdw.eq(dxdxwArr[start][end])
					.sum(Integer.MAX_VALUE).getFloat(0, 0), 1e-1);
		}

		for (int diff = 2; diff <= length; diff++) {
			for (int start = 0; start + diff <= length; start++) {
				int end = start + diff;
				for (int split = start + 1; split < end; split++) {
					INDArray truedxdw = eye;
					assertEquals(dim * dim,
						truedxdw.eq(dxdxwArr[start][end])
							.sum(Integer.MAX_VALUE).getFloat(0, 0), 1e-1);
				}
			}
		}
	}
}
