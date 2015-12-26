package com.kushalarora.test.optimizer;

import com.kushalarora.compositionalLM.derivatives.IDerivatives;
import com.kushalarora.compositionalLM.model.CompositionalInsideOutsideScore;
import com.kushalarora.compositionalLM.model.IParameter;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.optimizer.AbstractOptimizer;
import com.kushalarora.compositionalLM.optimizer.IIndexedSized;
import com.kushalarora.compositionalLM.options.Options;
import org.apache.commons.configuration.ConfigurationException;
import org.jblas.util.Random;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;

/**
 * Created by arorak on 12/26/15.
 */
public class AbstractOptimizerTest {
    private class Input extends ArrayList<Double> implements IIndexedSized {

        public long getSize()
        {
            return size();
        }

        public int getIndex()
        {
            return 0;
        }
    }

    private class Derivatives implements IDerivatives<Input> {

        public void add(IDerivatives<Input> derivatives)
        {

        }

        public void mul(double learningRate)
        {

        }

        public void calcDerivative(Model model, CompositionalInsideOutsideScore scorer)
        {

        }

        public void clear()
        {

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

    private AbstractOptimizer<Input, Derivatives> abstractOptimizer;

    @Before
    public void setUp() throws ConfigurationException {
        Options op = new Options();
        op.trainOp.parallel = false;
        abstractOptimizer = new AbstractOptimizer<Input, Derivatives>(op, new Derivatives())
        {
            public Derivatives calcDerivative(Input sample)
            {
                return null;
            }

            public void updateParams(Derivatives derivatives)
            {

            }

            public IParameter getParams()
            {
                return null;
            }

            public void clearLearningRate()
            {

            }

            public void calcLearningRate(Derivatives derivatives)
            {

            }

            public double getValidationScore(Input data)
            {
                return 0;
            }

            public void saveModel(int iter, int epoch)
            {

            }
        };
    }

    @Test
    public void testAbstractOptimizer()
    {
        Input input = new Input();
        for (int i = 0; i < 20; i++) {
            input.add(3.14 * Random.nextDouble());
        }


    }
}
