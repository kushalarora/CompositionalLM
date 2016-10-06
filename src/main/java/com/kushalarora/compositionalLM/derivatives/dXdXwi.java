package com.kushalarora.compositionalLM.derivatives;

import javax.annotation.Nullable;

import com.kushalarora.compositionalLM.lang.StanfordCompositionalInsideOutsideScore;
import com.kushalarora.compositionalLM.optimizer.IIndexedSized;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.google.common.base.Function;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.options.Options;
import com.kushalarora.compositionalLM.utils.Parallelizer;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;

/**
 * Created by karora on 6/21/15.
 */

@Slf4j
public class dXdXwi<T extends IIndexedSized> {
	@Getter
	private INDArray[][] dXdXwl;
	private int dim;
	private int V;
	private T data;
	private int length;
	private Options op;
	private Parallelizer parallelizer;
	private int i;

	public dXdXwi(int dimension, int vocab, T data, Options op, int i) {
		dim = dimension;
		V = vocab;
		this.data = data;
		length = data.getSize();
		dXdXwl = new INDArray[length][length + 1];
		this.op = op;
		this.i = i;
		parallelizer = new Parallelizer(op, op.grammarOp.maxLength / op.trainOp.blockNum + 1);
	}

	public INDArray[][] calcDerivative(final Model model,
	                                   final StanfordCompositionalInsideOutsideScore scorer) {

		final INDArray[][] phraseMatrix = scorer.getPhraseMatrix();
		final double[][][] compositionISplitScore = scorer.getCompISplitScore();
		final double[][] compositionIScore = scorer.getCompIScores();

		// dw/dxi = 0 if w != xi
		for (int start = 0; start < length; start++) {
			int end = start + 1;
			dXdXwl[start][end] = Nd4j.zeros(dim, dim);
		}

		// dw/dxi = I_n if w = x_i
		dXdXwl[i][i + 1] = Nd4j.eye(dim);

		for (int diff = 2; diff <= length; diff++) {
			for (int st = 0; st + diff <= length; st++) {
				final int start = st;
				final int end = start + diff;
				dXdXwl[start][end] = Nd4j.zeros(dim, dim);

				if (compositionIScore[start][end] == 0) {
					continue;
				}

				Function<Integer, Void> splitFunc = new Function<Integer, Void>() {
					@Nullable
					public Void apply(@Nullable Integer split) {
						// Calculate f'(c_1, c_2)
						INDArray child1 = phraseMatrix[start][split];
						INDArray child2 = phraseMatrix[split][end];
						INDArray dC = model.composeDerivative(child1, child2);
						dC = dC.transpose().broadcast(new int[]{dim, dim}).transpose();

						// [dc_1dW_ij dc_2dW_ij].transpose()
						INDArray dC12 = Nd4j.concat(0, dXdXwl[start][split], dXdXwl[split][end]);

						// f'(c1, c2) \circ
						dC = dC.muli(
							// W *
							model
								.getParams()
								.getW()
								// [dc_1 dc_2]^T))
								.mmul(dC12));


						// weighted marginalization over split
						synchronized (dXdXwl[start][end]) {

							double splitNorm =
								compositionISplitScore[start][end][split]/compositionIScore[start][end];

							dXdXwl[start][end] =
								dXdXwl[start][end]
									.addi(
										dC
										// \pi[start][end][split]
										.muli(splitNorm));
						}
						return null;
					}
				};

				if (op.trainOp.modelParallel) {
					parallelizer.parallelizer(start + 1, end, splitFunc);
				} else {
					for (int split = start + 1; split < end; split++) {
						splitFunc.apply(split);
					}
				}


			}
		}
		return dXdXwl;
	}
}
