package com.kushalarora.compositionalLM.model;

import java.util.Map;

import com.kushalarora.compositionalLM.derivatives.Derivatives;
import com.kushalarora.compositionalLM.derivatives.IDerivatives;
import com.kushalarora.compositionalLM.lang.Sentence;
import com.kushalarora.compositionalLM.options.Options;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


/**
 * Created by karora on 7/14/15.
 */

@Getter
@Setter
@Slf4j
public class Parameters implements IParameter<Sentence> {
    private final Options op;
    private INDArray W;
    private INDArray u;
    private INDArray h1;
    private INDArray h2;
    private INDArray X;
    private final int dimensions;
    private final int vocabSize;

    public Parameters(Options op, int dimensions, int vocabSize) {
        RandomGenerator rng = new JDKRandomGenerator();
        rng.setSeed(2204);
        this.dimensions = dimensions;
        this.vocabSize = vocabSize;
        W = Nd4j.rand(dimensions, 2 * dimensions);      // d X 2d matrix
        u = Nd4j.rand(dimensions, 1);                   // row vector with d entries
        h1 = Nd4j.rand(dimensions, 1);                  // row vector with d entries
        h2 = Nd4j.rand(dimensions, 1);                  // row vector with d entries
        X = Nd4j.rand(vocabSize, dimensions);           // V X d matrix
        this.op = op;
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
        if (h1 != null ?
                h1.neq(that.h1).sum(Integer.MAX_VALUE).getFloat(0) != 0 :
                that.h1 != null)
            return false;
        if (h2 != null ?
                h2.neq(that.h2).sum(Integer.MAX_VALUE).getFloat(0) != 0 :
                that.h2 != null)
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
        result = 31 * result + (h1 != null ? h1.hashCode() : 0);
        result = 31 * result + (h2 != null ? h2.hashCode() : 0);
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
        W.addi(dq.getDqdw().getDQdW());
        log.info("new W =\n {}", W);

        log.info("old u = \n {}", u);
        log.info("du = \n {}", dq.getDqdu().getDQdu());
        u.addi(dq.getDqdu().getDQdu());
        log.info("new u = \n {}", u);

	    log.info("dh1 = \n {}", dq.getDqdh1().getDQdh1());
	    h1.addi(dq.getDqdh1().getDQdh1());
	    log.info("new h1 = \n {}", h1);

	    log.info("dh2 = \n {}", dq.getDqdh2().getDQdh2());
	    h2.addi(dq.getDqdh2().getDQdh2());
	    log.info("new h2 = \n {}", h2);

        log.info("dX = \n {}", dq.getDqdxw().getIndexToxMap());

        for (Map.Entry<Integer, INDArray> entry :
                ((Map<Integer, INDArray>)dq.getDqdxw().getIndexToxMap()).entrySet()) {
            Integer key = entry.getKey();
            INDArray value = entry.getValue();
            X.putRow(key, value.transpose().addi(X.getRow(key)));
        }

        double l2term = op.trainOp.l2term;
        if (l2term != 0) {
            u.subi(u.mul(l2term));
            W.subi(W.mul(l2term));
            X.subi(X.mul(l2term));
	        h1.subi(h1.mul(l2term));
	        h2.subi(h2.mul(l2term));
        }

	    if (op.debug) {
		    log.info("$#Norm2 u : {}", Nd4j.norm2(u));
		    log.info("$#Norm2 W : {}", Nd4j.norm2(W));
		    log.info("$#Norm2 X : {}", Nd4j.norm2(X));
	    }
    }
}
