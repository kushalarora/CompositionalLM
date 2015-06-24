package com.kushalarora.compositionalLM.model;

import com.kushalarora.compositionalLM.lang.Word;
import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.Serializable;
import java.lang.reflect.Method;

/**
 * Model class for compositional LM.
 * Created by karora on 6/17/15.
 */
@Data
@AllArgsConstructor
@Slf4j
public class Parameters implements Serializable {
    public enum Activation {
        SIGMOID("sigmoid"),
        LINEAR("identity"),
        TANH("tanh");


        private final String text;

        private Activation(final String text) {
            this.text = text;
        }

        @Override
        public String toString() {
            return text;
        }
    }

    private int dimensions;
    private int vocabSize;
    @Setter(AccessLevel.PACKAGE)
    private INDArray W;
    @Setter(AccessLevel.PACKAGE)
    private INDArray u;
    @Setter(AccessLevel.PACKAGE)
    private INDArray X;
    private Activation f;
    private Activation g;

    public Parameters(int dimensions, int vocabSize, Activation composition, Activation output) {
        this.dimensions = dimensions;
        this.vocabSize = vocabSize;
        W = Nd4j.create(dimensions, 2 * dimensions);    // d X 2d matrix
        u = Nd4j.create(1, dimensions);                // row vector with d entries
        X = Nd4j.create(dimensions, vocabSize);         // d X V matrix
        f = composition;                                // default composition activation
        g = output;                         // default output activation
    }

    public Parameters(int dimensions, int vocabSize) {
        this(dimensions, vocabSize, Activation.TANH, Activation.LINEAR);
    }

    public Parameters(@NonNull INDArray X, @NonNull Activation composition, @NonNull Activation output) {
        this.dimensions = X.size(0);
        this.vocabSize = X.size(1);
        this.X = X;
        W = Nd4j.create(dimensions, 2 * dimensions);    // d X 2d matrix
        u = Nd4j.create(dimensions, 1);                 // row vector with d entries
        f = composition;                                // default composition activation
        g = output;                                    // default output activation
    }

    public Parameters(@NonNull INDArray X) {
        this(X, Activation.TANH, Activation.LINEAR);
    }
}
