package com.kushalarora.compositionalLM.model;

import com.kushalarora.compositionalLM.lang.Word;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import lombok.NonNull;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import lombok.val;

import static com.kushalarora.compositionalLM.model.Parameters.Activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.Serializable;
import java.lang.reflect.Method;

/**
 * Created by karora on 6/18/15.
 */
@Slf4j
public class Model implements Serializable {
    Parameters params;

    public Model(Parameters params, LexicalizedParser parserGrammar) {
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
        if (index < 0 || index >= params.getVocabSize()) {
            throw new RuntimeException(String.format("Word index must be between 0 to %d. " +
                    "Word::Index %s::%d", params.getVocabSize(), word.toString(), word.getIndex()));
        }
        return params.getX().getColumn(index);
    }

    @SneakyThrows
    private static INDArray applyActivation(INDArray arr,
                                            Activation activation) {
        Method method = null;
        try {
            method = Transforms.class.getDeclaredMethod(activation.toString(), INDArray.class);
        } catch (NoSuchMethodException e) {
            log.error("Activation {} not found.", activation);
            throw e;
        }
        assert method != null;
        return (INDArray) method.invoke(null, arr);
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
        } else if (child1.size(0) != params.getDimensions() ||
                child2.size(0) != params.getDimensions()) {
            throw new IllegalArgumentException(String.format("Child1 and Child2 should of size %d. " +
                            "Current sizes are  : (%d, %d)", params.getDimensions(),
                    child1.size(0), child2.size(0)));
        }
        val child12 = Nd4j.concat(0, child1, child2);
        return applyActivation(params.getW().mmul(child12), params.getF());
    }

    /**
     * Given a node and both the children compute the composition energy for the node.
     *
     * @param node   Parent node embedding. d dimension column vector
     * @param child1 left child embedding. d dimension column vector
     * @param child2 right child embedding. d dimension column vector
     * @return energy value for the composition.
     */
    public INDArray energy(@NonNull INDArray node, INDArray child1, INDArray child2) {
        if (!node.isColumnVector()) {
            throw new RuntimeException("Composed node should be a column vector");
        } else if (node.size(0) != params.getDimensions()) {
            throw new IllegalArgumentException(String.format("Node should of size %d. " +
                    "Current size is: (%d)", params.getDimensions(), node.size(0)));
        }
        return applyActivation(params.getU().mmul(node), params.getG());
    }

    /**
     * Compute energy for the leaf node where there are no children
     *
     * @param node Leaf node embedding. d dimension column vector.
     * @return energy value for the leaf node.
     */
    public INDArray energy(@NonNull INDArray node) {
        return this.energy(node, null, null);
    }
}
