package com.kushalarora.compositionalLM.model;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by karora on 7/14/15.
 */

@Getter
@Setter
public class Parameters implements IParameter {
    private INDArray W;
    private INDArray u;
    private INDArray X;
    private final int dimensions;
    private final int vocabSize;

    public Parameters(int dimensions, int vocabSize) {
        this.dimensions = dimensions;
        this.vocabSize = vocabSize;
        W = Nd4j.rand(dimensions, 2 * dimensions);      // d X 2d matrix
        u = Nd4j.rand(1, dimensions);                   // row vector with d entries
        X = Nd4j.rand(dimensions, vocabSize);           // d X V matrix
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

    public void update(IParameter params) {
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

    public void update(IParameterDerivatives derivatives) {
        Derivatives dq = (Derivatives) derivatives;
        W.add(dq.getDqdw().getDQdW());
        u.add(dq.getDqdu().getDQdu());
        X.add(dq.getDqdxw().getDQdXw());
    }
}
