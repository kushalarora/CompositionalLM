package com.kushalarora.compositionalLM.model;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.activation.ActivationFunction;
import org.nd4j.linalg.api.activation.Activations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;

/**
 * Model class for compositional LM.
 * Created by karora on 6/17/15.
 */
@Slf4j
@Getter
public class Parameters implements Serializable {
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

    public Parameters(int dimensions, int vocabSize, ActivationFunction composition, ActivationFunction output) {
        this.dimensions = dimensions;
        this.vocabSize = vocabSize;
        W = Nd4j.rand(dimensions, 2 * dimensions);    // d X 2d matrix
        u = Nd4j.rand(1, dimensions);                // row vector with d entries
        X = Nd4j.rand(dimensions, vocabSize);         // d X V matrix
        f = composition;                                // default composition activation
        g = output;                         // default output activation
    }

    public Parameters(int dimensions, int vocabSize) {
        this(dimensions, vocabSize, Activations.tanh(), Activations.linear());
    }

    public Parameters(@NonNull INDArray X, @NonNull ActivationFunction composition, @NonNull ActivationFunction output) {
        this.dimensions = X.size(0);
        this.vocabSize = X.size(1);
        this.X = X;
        W = Nd4j.create(dimensions, 2 * dimensions);    // d X 2d matrix
        u = Nd4j.create(dimensions, 1);                 // row vector with d entries
        f = composition;                                // default composition activation
        g = output;                                    // default output activation
    }

    public Parameters(@NonNull INDArray X) {
        this(X, Activations.hardTanh(), Activations.linear());
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Parameters that = (Parameters) o;

        if (dimensions != that.dimensions) return false;
        if (vocabSize != that.vocabSize) return false;
        if (W != null ? W.neq(that.W).sum(Integer.MAX_VALUE).getFloat(0) != 0 : that.W != null) return false;
        if (u != null ? u.neq(that.u).sum(Integer.MAX_VALUE).getFloat(0) != 0 : that.u != null) return false;
        if (X != null ? X.neq(that.X).sum(Integer.MAX_VALUE).getFloat(0) != 0 : that.X != null) return false;
        if (f != null ? !f.getClass().equals(that.f.getClass()) : that.f != null) return false;
        return !(g != null ? !g.getClass().equals(that.g.getClass()) : that.g != null);
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
