package com.kushalarora.compositionalLM.model;

import com.kushalarora.compositionalLM.derivatives.dQdW;
import com.kushalarora.compositionalLM.derivatives.dQdXw;
import com.kushalarora.compositionalLM.derivatives.dQdu;
import com.kushalarora.compositionalLM.lang.IGrammar;
import com.kushalarora.compositionalLM.lang.Word;
import lombok.Getter;
import lombok.NonNull;
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
    private INDArray W;
    private dQdW dqdw;
    private INDArray u;
    private dQdu dqdu;
    private INDArray X;
    private dQdXw dqdxw;
    private ActivationFunction f;
    private ActivationFunction g;
    private IGrammar grammar;

    public Model(@NonNull int dimensions,
                 @NonNull IGrammar iGrammar,
                 @NonNull ActivationFunction composition,
                 @NonNull ActivationFunction output) {

        this.grammar = iGrammar;
        this.dimensions = dimensions;
        this.vocabSize = iGrammar.getVocabSize();


        W = Nd4j.rand(dimensions, 2 * dimensions);      // d X 2d matrix
        u = Nd4j.rand(1, dimensions);                   // row vector with d entries
        X = Nd4j.rand(dimensions, vocabSize);           // d X V matrix
        f = composition;                                // default composition activation
        g = output;                                     // default output activation

        // IMPORTANT::The order must be preserved here
        // all derivatives should be the last one to be
        // initialized
        dqdu = new dQdu(this);
        dqdw = new dQdW(this);
        dqdxw = new dQdXw(this);
    }

    public Model(@NonNull int dimensions,
                 @NonNull IGrammar iGrammar) {
        this(dimensions, iGrammar, Activations.tanh(), Activations.linear());
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
        int[] valShape = valObj.shape();
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
        int[] valShape = valObj.shape();
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

    /**
     * Calculate derivative and update the parameter W,u and X.
     *
     * @param learningRate Learning rate for update
     * @param scorer       Compositional scores used in calculating derivatives
     */
    public void update(double learningRate, CompositionalGrammar.CompositionalInsideOutsideScorer scorer) {
        W = W.sub(
                dqdw.calcDerivative(scorer)
                        .mul(learningRate));
        u = u.sub(
                dqdu.calcDerivative(scorer)
                        .mul(learningRate));
        X = X.sub(
                dqdxw.calcDerivative(scorer)
                        .mul(learningRate));
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
