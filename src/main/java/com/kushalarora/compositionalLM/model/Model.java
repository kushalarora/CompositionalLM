package com.kushalarora.compositionalLM.model;

import com.google.common.base.Function;
import com.kushalarora.compositionalLM.lang.GrammarFactory;
import com.kushalarora.compositionalLM.lang.Word;
import com.kushalarora.compositionalLM.options.Options;
import java.util.Set;
import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.ops.impl.transforms.Identity;
import org.nd4j.linalg.api.ops.impl.transforms.Sigmoid;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;

import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;


// TODO: Try to remove code duplication from derivatives. Use forward pass instead.
@Slf4j
@Getter
public class Model implements Serializable {

	private int dimensions;
	private int vocabSize;
	Parameters params;
	private GrammarFactory.GrammarType grammarType;
	private double ZWord;
	private Set<Word> vocab;
	private int grammarVocabSize;

	public Model(@NonNull Options op,
	             @NonNull int dimensions,
	             @NonNull int vocabSize,
	             @NonNull GrammarFactory.GrammarType grammarType) {

		this.grammarType = grammarType;
		this.dimensions = dimensions;
		this.vocabSize = vocabSize;
		this.params = new Parameters(op, dimensions, vocabSize);
	}

	public Model(@NonNull Parameters params, @NonNull GrammarFactory.GrammarType grammarType) {
		this.grammarType = grammarType;
		this.dimensions = params.getDimensions();
		this.params = params;
		this.grammarVocabSize = params.getGrammarVocabSize();
	}

	public void setVocab(Set<Word> vocab) {
		this.vocab = vocab;
		vocabSize = vocab.size();
	}

	/**
	 * Returns the continuous space embedding of the word
	 *
	 * @param word Queried word.
	 * @return d dimension embedding of the word
	 */
	public INDArray word2vec(@NonNull Word word) {
		int index = word.getIndex();
		int grammarVocabSize = params.getGrammarVocabSize();
		if (index < 0 || index >= grammarVocabSize) {
			throw new RuntimeException(String.format("Word index must be between 0 to %d. " +
				"Word::Index %s::%d", params.getGrammarVocabSize(), word.toString(), word.getIndex()));
		}
		return params.getX().getRow(index).transpose();
	}

	/**
	 * Compose parent node from two children.
	 *
	 * @param child1 left child embedding. d dimension column vector.
	 * @param child2 right child embedding. d dimension column vector.
	 * @return return  continuous vector representation of parent node. d dimension column vector
	 */
	public INDArray compose(@NonNull INDArray child1, @NonNull INDArray child2) {
		if (!child1.isColumnVector() || !child2.isColumnVector()) {
			throw new IllegalArgumentException("Child1 and Child2 should be column vectors");
		} else if (child1.size(0) != dimensions || child2.size(0) != dimensions) {
			throw new IllegalArgumentException(
				String.format("Child1 and Child2 should of size %d. " +
					"Current sizes are  : (%d, %d)", dimensions, child1.size(0), child2.size(0)));
		}
		INDArray child12 = Nd4j.concat(0, child1, child2);
		return exec(new Sigmoid(params.getW().mmul(child12)));
	}


	public INDArray composeDerivative(@NonNull INDArray child1, @NonNull INDArray child2) {
		if (!child1.isColumnVector() || !child2.isColumnVector()) {
			throw new IllegalArgumentException("Child1 and Child2 should be column vectors");
		} else if (child1.size(0) != dimensions ||
			child2.size(0) != dimensions) {
			throw new IllegalArgumentException(String.format("Child1 and Child2 should of size %d. " +
					"Current sizes are  : (%d, %d)", dimensions,
				child1.size(0), child2.size(0)));
		}
		INDArray child12 = Nd4j.concat(0, child1, child2);
		return exec(new Sigmoid(params.getW().mmul(child12)).derivative());
	}

	public INDArray linearWord(@NonNull INDArray node) {
		if (node.size(0) != dimensions) {
			throw new IllegalArgumentException(String.format(
				"Node should of size %d. " +
					"Current size is :(%d)",
				dimensions, node.size(0)));
		}

		return params.getU().transpose().mmul(node);
	}

	public double energyWord(@NonNull INDArray node) {
		INDArray valObj = exec(new Identity(linearWord(node)));
		int[] valShape = valObj.shape();
		if (valShape[0] != 1 && valShape[1] != 1) {
			throw new RuntimeException(
				"Expected a 1 X 1 matrix. Got " + valObj.shape().toString());
		}

		return valObj.getDouble(0);
	}

	/**
	 * Compute energy for the leaf node where there are no children
	 *
	 * @param node Leaf node embedding. d dimension column vector.
	 * @return energy value for the leaf node.
	 */
	public double unProbabilityWord(@NonNull INDArray node) {

		return Math.exp(-energyWord(node));
	}

	public double probabilityWord(@NonNull INDArray node) {

		double unProb = unProbabilityWord(node);
		if (ZWord == 0) {
			calculateZ();
		}
		return unProb/ZWord;
	}

	public double energyWordDerivative(@NonNull INDArray node) {

		INDArray valObj = exec(new Identity(linearWord(node)).derivative());
		int[] valShape = valObj.shape();
		if (valShape[0] != 1 && valShape[1] != 1) {
			throw new RuntimeException(
				"Expected a 1 X 1 matrix. Got " + valObj.shape().toString());
		}
		return valObj.getDouble(0);
	}

	public INDArray linearComposition(@NonNull INDArray node, INDArray child1, INDArray child2) {
		if (node.size(0) != dimensions ||
			child1.size(0) != dimensions ||
			child2.size(0) != dimensions) {
			throw new IllegalArgumentException(String.format("Node, child1 and child2 should of size %d. " +
					"Current size are: Node:(%d), child1:(%s), child2:(%s)",
				dimensions, node.size(0), child1.size(0), child2.size(0)));
		}

		return params.getU().transpose().mmul(node)
			.addi(params.getH1().transpose().mmul(child1))
			.addi(params.getH2().transpose().mmul(child2));
	}

	public double energyComp(@NonNull INDArray node, INDArray child1, INDArray child2) {
		INDArray valObj = exec(new Identity(linearComposition(node, child1, child2)));

		int[] valShape = valObj.shape();
		if (valShape[0] != 1 && valShape[1] != 1) {
			throw new RuntimeException("Expected a 1 X 1 matrix. Got " + valObj.shape().toString());
		}

		return valObj.getDouble(0);
	}

	/**
	 * Given a node and both the children compute the composition energy for the node.
	 *
	 * @param node   Parent node embedding. d dimension column vector
	 * @param child1 left child embedding. d dimension column vector
	 * @param child2 right child embedding. d dimension column vector
	 * @return energy value for the composition.
	 */
	public double unProbabilityComp(@NonNull INDArray node, INDArray child1, INDArray child2) {
		return Math.exp(-energyComp(node, child1, child2));
	}

	public double energyCompDerivative(@NonNull INDArray node, INDArray child1, INDArray child2) {

		INDArray valObj =
			exec(new Identity(linearComposition(node, child1, child2))
					.derivative());

		int[] valShape = valObj.shape();
		if (valShape[0] != 1 && valShape[1] != 1) {
			throw new RuntimeException("Expected a 1 X 1 matrix. Got " + valShape.toString());
		}
		return valObj.getDouble(0);
	}

	double  calculateZ() {
		log.info("Calculating Z_Word.");
		if (vocab == null) {
			throw new RuntimeException(
				"Vocab should be set before running the model");
		}
		for (Word word: vocab) {
			INDArray x = word2vec(word);
			ZWord += unProbabilityWord(x);
		}
		return ZWord;
	}

	public void preProcessOnBatch() {
		calculateZ();
	}

	public void postProcessOnBatch() {
		log.info("Cleaning up ZWord");
		ZWord = 0;
	}

	public INDArray Expectedl(int i, int j,
	                          INDArray[] E_ij,
	                          INDArray[] compositionMatrices,
	                          INDArray[][] phraseMatrix,
	                          double compMuSum,
	                          int[] dims) {
		double Zl = 0;
		INDArray E_l = Nd4j.zeros(dims);
		for (int s = i + 1; s < j; s++) {
			double Zls = unProbabilityComp(
								compositionMatrices[s],
								phraseMatrix[i][s],
								phraseMatrix[s][j]);
			E_l.addi(E_ij[s].muli(Zls));
			Zl += Zls;
		}
		return E_l.muli(compMuSum).divi(Zl);
	}

	public INDArray ExpectedV(Function<INDArray, INDArray> wTodELeafFunc, int[] dims) {

		INDArray EV = Nd4j.zeros(dims);
		if (vocab == null) {
			throw new RuntimeException(
						"Vocab must be set before running the model.");
		}
		for (Word word : vocab) {
			INDArray x = word2vec(word);
			INDArray xD = wTodELeafFunc.apply(x);
			xD.muli(probabilityWord(x));
			EV.addi(xD);
		}
		return EV;
	}

	@Override
	public boolean equals(Object o) {
		if (this == o) return true;
		if (o == null || getClass() != o.getClass()) return false;

		Model that = (Model) o;
		if (dimensions != that.dimensions) return false;
		if (vocabSize != that.vocabSize) return false;
		if (!params.equals(params)) return false;
		return true;
	}

	@Override
	public int hashCode() {
		int result = dimensions;
		result = 31 * result + vocabSize;
		result = 31 * result + (params != null ? params.hashCode() : 0);
		return result;
	}

	private static INDArray exec(TransformOp op) {
		if (op.x().isCleanedUp()) {
			throw new IllegalStateException("NDArray already freed");
		} else {
			return Nd4j.getExecutioner().execAndReturn(op);
		}
	}
}
