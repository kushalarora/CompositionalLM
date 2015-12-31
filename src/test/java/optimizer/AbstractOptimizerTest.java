package optimizer;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import com.kushalarora.compositionalLM.derivatives.IDerivatives;
import com.kushalarora.compositionalLM.lang.Word;
import com.kushalarora.compositionalLM.model.IParameter;
import com.kushalarora.compositionalLM.optimizer.AbstractOptimizer;
import com.kushalarora.compositionalLM.optimizer.AbstractSGDOptimizer;
import com.kushalarora.compositionalLM.optimizer.IIndexed;
import com.kushalarora.compositionalLM.optimizer.IIndexedSized;
import com.kushalarora.compositionalLM.options.Options;
import lombok.Getter;
import org.apache.commons.configuration.ConfigurationException;
import org.jblas.util.Random;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.ExecutionException;

public class AbstractOptimizerTest {
/*
    public class InputWord extends Word  {

        public InputWord(String word, int index) {
            super(word, index);
        }

        public int getIndex() {
            return 0;
        }
    }

    public class Input extends ArrayList<Integer> implements IIndexedSized {
        private int index;
        public Input(int index) {
            this.index = index;
        }

        public int getSize() {
            return size();
        }

        public int getIndex() {
            return index;
        }

        @Override
        public IIndexed get(int index) {
            return get(index);
        }
    };

    public class Derivatives implements IDerivatives<Input> {
        Function<Double, Double> derivativeFunc;
        Parameters parameters;
        @Getter
        List<Double> derivatives;
        public Derivatives(Function<Double, Double> derivativeFunc, Parameters parameters) {
            this.derivativeFunc = derivativeFunc;
            this.parameters = parameters;
            derivatives = new ArrayList<Double>();
        }

        public Derivatives(Function<Double, Double> derivativeFunc) {
            this.derivativeFunc = derivativeFunc;
            derivatives = new ArrayList<Double>();
        }

        public void add(IDerivatives<Input> iderivatives) {
            Derivatives dderivatives = (Derivatives)iderivatives;
            for (int i =0; i < dderivatives.derivatives.size(); i++) {
                derivatives.set(i, derivatives.get(i) + dderivatives.derivatives.get(i));
            }
        }

        public void mul(double learningRate) {
            for (int i = 0; i < derivatives.size(); i++) {
                derivatives.set(i, derivatives.get(i) * learningRate);
            }
        }

        public void calcDerivative() {
            for (int i = 0; i < parameters.xs.size(); i++) {
                derivatives.add(derivativeFunc.apply(parameters.xs.get(i)));
            }
        }

        public void clear() {
            derivatives.clear();
        }

        public Input getData()
        {
            return null;
        }

        public IDerivatives<Input> adaGrad(IDerivatives<Input> derivatives)
        {
            return null;
        }

        public double getScore()
        {
            return 0;
        }
    }

    private class Parameters implements IParameter<Input> {
        List<Double> xs;
        public Parameters() {
            xs = new ArrayList<Double>();
            for (int i = 0; i < 20; i++) {
                xs.add(Random.nextDouble());
            }
        }
        public void update(IDerivatives<Input> derivatives) {

            Derivatives dderivatives = (Derivatives)derivatives;
            for (int i = 0; i < xs.size(); i++) {
                List<Double> derivativeValues = dderivatives.derivatives;
                xs.set(i, xs.get(i) + derivativeValues.get(i));
            }
        }
    }



    public AbstractOptimizer<Input, Derivatives> abstractOptimizer;


    @Test
    public void testOptimzer() throws ConfigurationException, ExecutionException, InterruptedException {
        final Parameters parameters = new Parameters();

        final Function<Double, Double> scorer = new Function<Double, Double>() {
            double a = 0.5;
            double b = 0.1;
            double c = 2.1;
            double dd = -0.2;
            public Double apply(Double d) {
                return Math.pow(
                        Math.sin(d) -
                                a * Math.pow(d, 3) +
                                b * Math.pow(d, 2) +
                                c * Math.pow(d, 1) +
                                dd,
                        2);
            }
        };

        final Function<Double, Double> derivativeFunc = new Function<Double, Double>() {
            double a = 0.5;
            double b = 0.1;
            double c = 2.1;
            public Double apply(Double d) {
                return 2 * (Math.sin(d) - scorer.apply(d)) * (
                        Math.cos(d) -
                        3 * a * Math.pow(d, 2) +
                        2 * b * Math.pow(d, 1) + c);
            }
        };


        Options op = new Options();
        op.trainOp.parallel = false;
        op.trainOp.learningRate = 1;
        op.trainOp.maxEpochs = 1;
        abstractOptimizer = new AbstractSGDOptimizer<Input, Derivatives>(op, new Derivatives(derivativeFunc)) {
            public Derivatives calcDerivative(Input sample) {
                Derivatives derivatives = new Derivatives(derivativeFunc, parameters);
                derivatives.calcDerivative();
                return derivatives;
            }

            public IParameter getParams() {
                return parameters;
            }

            public double getValidationScore(Input data) {
                double score = 0;
                for (int i = 0; i < parameters.xs.size(); i++) {
                    score += scorer.apply(parameters.xs.get(i));
                }
                return score;
            }

            public void saveModel(int iter, int epoch) {

            }
        };

        List<List<Input>> listOfList = new ArrayList<List<Input>>();
        List<Input> inputList = Lists.newArrayList(new Input(1), new Input(2), new Input(3));
        listOfList.add(inputList);

        abstractOptimizer.fit(listOfList, listOfList);
    }
    */
}