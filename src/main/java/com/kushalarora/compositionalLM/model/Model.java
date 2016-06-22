package com.kushalarora.compositionalLM.model;

import com.kushalarora.compositionalLM.lang.GrammarFactory;
import com.kushalarora.compositionalLM.lang.Word;
import com.kushalarora.compositionalLM.options.Options;
import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.ops.impl.transforms.Identity;
import org.nd4j.linalg.api.ops.impl.transforms.Sigmoid;
import org.nd4j.linalg.api.ops.impl.transforms.Tanh;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;

/**
 * Created by karora on 6/18/15.
 */
@Slf4j
@Getter
public class Model implements Serializable {

    private int dimensions;
    private int vocabSize;
    Parameters params;
    private GrammarFactory.GrammarType grammarType;

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
        this.vocabSize = params.getVocabSize();
        this.params = params;
    }

    /**
     * Returns the continuous space embedding of the word
     *
     * @param word Queried word.
     * @return d dimension embedding of the word
     */
    public INDArray word2vec(@NonNull Word word) {
        int index = word.getIndex();
        if (index < 0 || index >= vocabSize) {
            throw new RuntimeException(String.format("Word index must be between 0 to %d. " +
                    "Word::Index %s::%d", vocabSize, word.toString(), word.getIndex()));
        }
        return params.getX().getColumn(index);
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
        } else if (child1.size(0) != dimensions ||
                child2.size(0) != dimensions) {
            throw new IllegalArgumentException(String.format("Child1 and Child2 should of size %d. " +
                            "Current sizes are  : (%d, %d)", dimensions,
                    child1.size(0), child2.size(0)));
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

    /**
     * Given a node and both the children compute the composition energy for the node.
     *
     * @param node   Parent node embedding. d dimension column vector
     * @param child1 left child embedding. d dimension column vector
     * @param child2 right child embedding. d dimension column vector
     * @return energy value for the composition.
     */
    public double energy(@NonNull INDArray node, INDArray child1, INDArray child2) {
        if (!node.isColumnVector()) {
            throw new RuntimeException("Composed node should be a column vector");
        } else if (node.size(0) != dimensions) {
            throw new IllegalArgumentException(String.format("Node should of size %d. " +
                    "Current size is: (%d)", dimensions, node.size(0)));
        }
        INDArray valObj = params.getU().mmul(node);
        int[] valShape = valObj.shape();
        if (valShape.length != 1 || valShape[0] != 1) {
            throw new RuntimeException("Expected a 1 X 1 matrix. Got " + valObj.shape().toString());
        }
        return valObj.getDouble(0);
    }


    public double energyDerivative(@NonNull INDArray node, INDArray child1, INDArray child2) {
        if (!node.isColumnVector()) {
            throw new RuntimeException("Composed node should be a column vector");
        } else if (node.size(0) != dimensions) {
            throw new IllegalArgumentException(String.format("Node should of size %d. " +
                    "Current size is: (%d)", dimensions, node.size(0)));
        }
        INDArray valObj = node;
        int[] valShape = valObj.shape();
        if (valShape.length != 1 || valShape[0] != 1) {
            throw new RuntimeException("Expected a 1 X 1 matrix. Got " + valShape.toString());
        }
        return valObj.getDouble(0);
    }

    public double energyDerivative(@NonNull INDArray node) {
        return energyDerivative(node, null, null);
    }

    /**
     * Compute energy for the leaf node where there are no children
     *
     * @param node Leaf node embedding. d dimension column vector.
     * @return energy value for the leaf node.
     */
    public double energy(@NonNull INDArray node) {
        return this.energy(node, null, null);
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
        if(op.x().isCleanedUp()) {
            throw new IllegalStateException("NDArray already freed");
        } else {
            return Nd4j.getExecutioner().execAndReturn(op);
        }
    }
}
