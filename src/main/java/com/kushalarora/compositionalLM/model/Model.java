package com.kushalarora.compositionalLM.model;

import com.kushalarora.compositionalLM.lang.Word;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.activation.ActivationFunction;
import org.nd4j.linalg.api.activation.Activations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;

/**
 * Created by karora on 6/18/15.
 */
@Slf4j
@Getter
public class Model implements Serializable {

    private int dimensions;
    private int vocabSize;
    @Setter(AccessLevel.PACKAGE)
    private INDArray W;
    @Setter(AccessLevel.PACKAGE)
    private INDArray u;
    @Setter(AccessLevel.PACKAGE)
    private INDArray X;
    private ActivationFunction f;
    private ActivationFunction g;

    public Model(int dimensions, int vocabSize, ActivationFunction composition, ActivationFunction output) {
        this.dimensions = dimensions;
        this.vocabSize = vocabSize;
        W = Nd4j.rand(dimensions, 2 * dimensions);    // d X 2d matrix
        u = Nd4j.rand(1, dimensions);                // row vector with d entries
        X = Nd4j.rand(dimensions, vocabSize);         // d X V matrix
        f = composition;                                // default composition activation
        g = output;                         // default output activation
    }

    public Model(int dimensions, int vocabSize) {
        this(dimensions, vocabSize, Activations.tanh(), Activations.linear());
    }

    public Model(@NonNull INDArray X, @NonNull ActivationFunction composition, @NonNull ActivationFunction output) {
        this.dimensions = X.size(0);
        this.vocabSize = X.size(1);
        this.X = X;
        W = Nd4j.create(dimensions, 2 * dimensions);    // d X 2d matrix
        u = Nd4j.create(dimensions, 1);                 // row vector with d entries
        f = composition;                                // default composition activation
        g = output;                                    // default output activation
    }

    public Model(@NonNull INDArray X) {
        this(X, Activations.hardTanh(), Activations.linear());
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
        return X.getColumn(index);
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
        return f.apply(W.mmul(child12));
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
        return f.applyDerivative(W.mmul(child12));
    }

    /**
     * Given a node and both the children compute the composition energy for the node.
     *
     * @param node   Parent node embedding. d dimension column vector
     * @param child1 left child embedding. d dimension column vector
     * @param child2 right child embedding. d dimension column vector
     * @return energy value for the composition.
     */
    public float energy(@NonNull INDArray node, INDArray child1, INDArray child2) {
        if (!node.isColumnVector()) {
            throw new RuntimeException("Composed node should be a column vector");
        } else if (node.size(0) != dimensions) {
            throw new IllegalArgumentException(String.format("Node should of size %d. " +
                    "Current size is: (%d)", dimensions, node.size(0)));
        }
        INDArray valObj = g.apply(u.mmul(node));
        int [] valShape = valObj.shape();
        if (valShape.length != 1 || valShape[0] != 1) {
            throw new RuntimeException("Expected a 1 X 1 matrix. Got " + valObj.shape().toString());
        }
        return valObj.getFloat(0);
    }


    public float energyDerivative(@NonNull INDArray node, INDArray child1, INDArray child2) {
        if (!node.isColumnVector()) {
            throw new RuntimeException("Composed node should be a column vector");
        } else if (node.size(0) != dimensions) {
            throw new IllegalArgumentException(String.format("Node should of size %d. " +
                    "Current size is: (%d)", dimensions, node.size(0)));
        }
        INDArray valObj = g.applyDerivative(u.mmul(node));
        int [] valShape = valObj.shape();
        if (valShape.length != 2 || valShape[0] != 1 || valShape[1] != 1) {
            throw new RuntimeException("Expected a 1 X 1 matrix. Got " + valObj.shape().toString());
        }
        return valObj.getFloat(0, 0);

    }

    public float energyDerivative(@NonNull INDArray node) {
        return energyDerivative(node, null, null);
    }

    /**
     * Compute energy for the leaf node where there are no children
     *
     * @param node Leaf node embedding. d dimension column vector.
     * @return energy value for the leaf node.
     */
    public float energy(@NonNull INDArray node) {
        return this.energy(node, null, null);
    }


    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Model that = (Model) o;

        if (dimensions != that.dimensions) return false;
        if (vocabSize != that.vocabSize) return false;
        if (W != null ?
                (W.neq(that.getW()).sum(Integer.MAX_VALUE).getFloat(0) != 0) :
                that.W != null)
            return false;
        if (u != null ?
                u.neq(that.u).sum(Integer.MAX_VALUE).getFloat(0) != 0 :
                that.u != null)
            return false;
        if (X != null ?
                X.neq(that.X).sum(Integer.MAX_VALUE).getFloat(0) != 0 :
                that.X != null)
            return false;
        if (f != null ?
                !f.getClass().equals(that.f.getClass()) :
                that.f != null)
            return false;
        return !(g != null ?
                !g.getClass().equals(that.g.getClass()) :
                that.g != null);
    }

    @Override
    public int hashCode() {
        int result = dimensions;
        result = 31 * result + vocabSize;
        result = 31 * result + (W != null ? W.hashCode() : 0);
        result = 31 * result + (u != null ? u.hashCode() : 0);
        result = 31 * result + (X != null ? X.hashCode() : 0);
        result = 31 * result + (f != null ? f.hashCode() : 0);
        result = 31 * result + (g != null ? g.hashCode() : 0);
        return result;
    }
}
