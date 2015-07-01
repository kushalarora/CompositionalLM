package com.kushalarora.compositionalLM.model;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.activation.ActivationFunction;
import org.nd4j.linalg.api.activation.Activations;
import org.nd4j.linalg.api.activation.HardTanh;
import org.nd4j.linalg.api.activation.Tanh;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import sun.rmi.server.Activation;

import java.io.Serializable;

/**
 * Model class for compositional LM.
 * Created by karora on 6/17/15.
 */
@Data
@AllArgsConstructor
@Slf4j
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
        W = Nd4j.create(dimensions, 2 * dimensions);    // d X 2d matrix
        u = Nd4j.create(1, dimensions);                // row vector with d entries
        X = Nd4j.create(dimensions, vocabSize);         // d X V matrix
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
}
