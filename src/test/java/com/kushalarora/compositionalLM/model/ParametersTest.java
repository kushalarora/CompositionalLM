package com.kushalarora.compositionalLM.model;

import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import lombok.val;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static edu.stanford.nlp.parser.lexparser.LexicalizedParser.loadModel;
import static org.junit.Assert.assertEquals;
import static org.nd4j.linalg.ops.transforms.Transforms.identity;
import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;

import static org.junit.Assert.assertTrue;

/**
 * Created by karora on 6/18/15.
 */
public class ParametersTest {

    private static Model model;

    @BeforeClass
    public static void setUpClass() {
        LexicalizedParser lp = loadModel();
        model = new Model(
                new Parameters(10, 100, Parameters.Activation.SIGMOID, Parameters
                .Activation
                .LINEAR), lp);
    }

    @Test
    public void testCompose() {
        val child1Vec = Nd4j.rand(10, 1);
        val child2Vec = Nd4j.rand(10, 1);
        val modelParentVec = model.compose(child1Vec, child2Vec);

        // verify it is a column vector
        assertTrue(modelParentVec.isColumnVector());

        // verify the shape is (10, 1)
        int[] shape = modelParentVec.shape();
        assertEquals(shape[0], 10);
        assertEquals(shape[1], 1);

        INDArray child12Vec = Nd4j.concat(0, child1Vec, child2Vec);
        INDArray trueParentVec = sigmoid(model.params.getW().mmul(child12Vec));
        assertEquals(String.format("True: %s,Model: %s", trueParentVec.toString(), modelParentVec.toString()), trueParentVec,
                modelParentVec);
    }

    @Test
    public void testLeafEnergy() {
        val vec = Nd4j.rand(10, 1);
        INDArray modelEnergy = model.energy(vec);

        val trueEnergy = identity(model.params.getU().mmul(vec));
        assertEquals(String.format("True: %s,Model: %s", trueEnergy.toString(), modelEnergy.toString()), trueEnergy,
                modelEnergy);
    }

    @Test
    public void testEnergy() {
        val child1Vec = Nd4j.rand(10, 1);
        val child2Vec = Nd4j.rand(10, 1);
        val parentVec = model.compose(child1Vec, child2Vec);
        INDArray modelEnergy = model.energy(parentVec, child1Vec, child2Vec);

        val trueEnergy = identity(model.params.getU().mmul(parentVec));
        assertEquals(String.format("True: %s,Model: %s", trueEnergy.toString(), modelEnergy.toString()),
                trueEnergy, modelEnergy);
    }

    @Test
    @Ignore

    public void testword2vec() {
        // TODO:: Complete the test after writing Word class
        assertTrue(false);
    }

    @Test(expected = RuntimeException.class)
    @Ignore

    public void testword2vecOutOfRangeException() {
        // TODO:: Complete the test after writing Word class
        assertTrue(false);
    }

    @Test(expected = RuntimeException.class)
    public void testcomposeChild1SizeException() {
        val vec10d = Nd4j.rand(10, 1);
        val vec12d = Nd4j.rand(12, 1);
        model.compose(vec12d, vec10d);
    }

    @Test(expected = RuntimeException.class)
    public void testcomposeChild2SizeException() {
        val vec10d = Nd4j.rand(10, 1);
        val vec12d = Nd4j.rand(12, 1);
        model.compose(vec10d, vec12d);
    }


    @Test(expected = RuntimeException.class)
    public void testcomposeChild1ShapeException() {
        val mat2d = Nd4j.rand(10, 2);
        val vec = Nd4j.rand(10, 1);
        model.compose(mat2d, vec);
    }

    @Test(expected = RuntimeException.class)
    public void testcomposeChild2ShapeException() {
        val mat2d = Nd4j.rand(10, 2);
        val vec = Nd4j.rand(10, 1);
        model.compose(vec, mat2d);
    }

    @Test(expected = RuntimeException.class)
    public void testenergySizeException() {
        val vec12d = Nd4j.rand(12, 1);
        model.energy(vec12d);
    }

    @Test(expected = RuntimeException.class)
    public void testenergyShapeException() {
        val mat2d = Nd4j.rand(10, 2);
        model.energy(mat2d);
    }
}
