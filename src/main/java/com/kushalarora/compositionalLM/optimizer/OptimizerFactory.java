package com.kushalarora.compositionalLM.optimizer;

import com.google.common.base.Function;
import com.kushalarora.compositionalLM.derivatives.Derivatives;
import com.kushalarora.compositionalLM.lang.IInsideOutsideScore;
import com.kushalarora.compositionalLM.model.IParameter;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.options.Options;
import com.kushalarora.compositionalLM.utils.Parallelizer;
import edu.stanford.nlp.util.IntTuple;

/**
 * Created by karora on 7/23/15.
 */
public class OptimizerFactory {
    public enum OptimizerType {
        SGD("sgd"),
        ADAGRAD("adagrad");

        private String text;

        OptimizerType(String text) {
            this.text = text;
        }

        public String getText() {
            return this.text;
        }

        public static OptimizerType fromString(String text) {
            if (text != null) {
                for (OptimizerType b : OptimizerType.values()) {
                    if (text.equalsIgnoreCase(b.text)) {
                        return b;
                    }
                }
            }
            return null;
        }
    }

    public static <T extends IIndexedSized>  AbstractOptimizer<T, Derivatives> getOptimizer(
            final Options op,
            final Model model,
            final Function<T, Double> trainScorer,
            final Function<T, Derivatives> derivativeCalculator,
            final Function<Void, Void> preProcessOnBatch,
            final Function<Void, Void> postProcessOnBatch,
            final Parallelizer parallelizer) {

        switch (op.trainOp.optimizer) {
            case SGD:
                return new AbstractSGDOptimizer<T, Derivatives>(op, new Derivatives(model, op), parallelizer) {
                    public Derivatives calcDerivative(T sample) {
                        return derivativeCalculator.apply(sample);
                    }

                    public IParameter getParams() {
                        return model.getParams();
                    }

                    public double getTrainScore(T data) {
                        return trainScorer.apply(data);
                    }

                    public void preProcessOnBatch() {
                        preProcessOnBatch.apply(null);
                    }

                    public void postProcessOnBatch() {
                        postProcessOnBatch.apply(null);
                    }
                };
            case ADAGRAD:
                return new AbstractAdaGradOptimzer<T, Derivatives>(
                        op, new Derivatives(model, op), new Derivatives(model, op), parallelizer) {
                    public Derivatives calcDerivative(T sample) {
                        return derivativeCalculator.apply(sample);
                    }

                    public IParameter getParams() {
                        return model.getParams();
                    }

                    public double getTrainScore(T data) {
                        return trainScorer.apply(data);
                    }
                    public void preProcessOnBatch() {
                        preProcessOnBatch.apply(null);
                    }

                    public void postProcessOnBatch() {
                        postProcessOnBatch.apply(null);
                    }
                };
            default:
                throw new RuntimeException("Could not find optimizer:"
                        + op.trainOp.optimizer.getText());
        }
    }
}
