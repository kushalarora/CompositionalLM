package com.kushalarora.compositionalLM.model;

import com.kushalarora.compositionalLM.lang.Sentence;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

/**
 * Created by karora on 7/14/15.
 */

@Getter
@Setter
@Slf4j
public class Parameters implements IParameter<Sentence> {
    private INDArray W;
    private INDArray u;
    private INDArray X;
    private final int dimensions;
    private final int vocabSize;

    public Parameters(int dimensions, int vocabSize) {
        Random random = new Random(1234);
        this.dimensions = dimensions;
        this.vocabSize = vocabSize;
        W = Nd4j.randn(dimensions, 2 * dimensions, 1234);      // d X 2d matrix
        u = Nd4j.randn(1, dimensions, 1234);                   // row vector with d entries
        X = Nd4j.randn(dimensions, vocabSize, 1234);           // d X V matrix
    }


    @Override
    public boolean equals(Object o) {
        if (this == o) return true;

        Parameters that = (Parameters) o;
        if (W != null ?
                (W.neq(that.W).sum(Integer.MAX_VALUE).getFloat(0) != 0) :
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
        return true;
    }

    public int hashCode() {
        int result = 0;
        result = 31 * result + (W != null ? W.hashCode() : 0);
        result = 31 * result + (u != null ? u.hashCode() : 0);
        result = 31 * result + (X != null ? X.hashCode() : 0);
        return result;
    }

    public void update(IParameter<Sentence> params) {
        Parameters parameters = (Parameters) params;
        if (dimensions != parameters.dimensions ||
                vocabSize != parameters.vocabSize) {
            new RuntimeException("parameter("
                    + dimensions + ", " + vocabSize + ") " +
                    "and updated parameter ("
                    + parameters.dimensions + ", " +
                    parameters.vocabSize + ") " +
                    "are not of same size");
        }
        W = parameters.W;
        u = parameters.u;
        X = parameters.X;
    }

    public void update(IDerivatives<Sentence> derivatives) {
        Derivatives dq = (Derivatives) derivatives;
        log.info("old W =\n {}", W);
        log.info("dW =\n {}", dq.getDqdw().getDQdW());
        W = W.add(dq.getDqdw().getDQdW());
        log.info("new W =\n {}", W);

        log.info("old u = \n {}", u);
        log.info("du = \n {}", dq.getDqdu().getDQdu());
        u = u.add(dq.getDqdu().getDQdu());
        log.info("new u = \n {}", u);

        log.info("old X = \n {}", X);
        log.info("dX = \n {}", dq.getDqdxw().getDQdXw());
        X = X.add(dq.getDqdxw().getDQdXw());
        log.info("new X = \n {}", X);
    }
}
