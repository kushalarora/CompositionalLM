package com.kushalarora.compositionalLM.optimizer;

import com.google.common.base.Function;
import com.kushalarora.compositionalLM.derivatives.Derivatives;
import com.kushalarora.compositionalLM.lang.Sentence;
import com.kushalarora.compositionalLM.model.IParameter;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.options.Options;
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

    public static AbstractOptimizer<Sentence, Derivatives> getOptimizer(
            final Options op,
            final Model model,
            final Function<Sentence, Double> scorer,
            final Function<Sentence, Derivatives> derivativeCalculator,
            final Function<IntTuple, Void> functionSaver) {

        int dimension = model.getDimensions();
        int vocabSize = model.getVocabSize();
        switch (op.trainOp.optimizer) {
            case SGD:
                return new AbstractSGDOptimizer<Sentence, Derivatives>(op, new Derivatives(dimension, vocabSize)) {
                    public Derivatives calcDerivative(Sentence sample) {
                        return derivativeCalculator.apply(sample);
                    }

                    public IParameter getParams() {
                        return model.getParams();
                    }

                    public double getValidationScore(Sentence data) {
                        return scorer.apply(data);
                    }

                    public void saveModel(int iter, int epoch) {
                        functionSaver.apply(new IntTuple(new int[] {iter, epoch}));
                    }
                };
            case ADAGRAD:
                return new AbstractAdaGradOptimzer<Sentence, Derivatives>(
                        op, new Derivatives(dimension, vocabSize), new Derivatives(dimension, vocabSize)) {

                    public Derivatives calcDerivative(Sentence sample) {
                        return derivativeCalculator.apply(sample);
                    }

                    public IParameter getParams() {
                        return model.getParams();
                    }

                    public double getValidationScore(Sentence data) {
                        return scorer.apply(data);
                    }

                    public void saveModel(int iter, int epoch) {
                        functionSaver.apply(new IntTuple(new int[] {iter, epoch}));
                    }
                };
            default:
                throw new RuntimeException("Could not find optimizer:"
                        + op.trainOp.optimizer.getText());
        }
    }
}
