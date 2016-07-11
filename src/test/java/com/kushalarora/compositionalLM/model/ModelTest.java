package com.kushalarora.compositionalLM.model;

import com.google.common.base.Function;
import com.kushalarora.compositionalLM.lang.GrammarFactory;
import com.kushalarora.compositionalLM.lang.IGrammar;
import com.kushalarora.compositionalLM.options.Options;
import com.kushalarora.compositionalLM.utils.Parallelizer;
import javax.annotation.Nullable;
import lombok.val;
import org.apache.commons.configuration.ConfigurationException;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.nd4j.linalg.ops.transforms.Transforms.identity;
import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;

public class ModelTest {

    private static Model model;
    private static int dim;
    @BeforeClass
    public static void setUpClass() throws ConfigurationException {
        Options op = new Options();
        op.grammarOp.grammarType = GrammarFactory.GrammarType.STANFORD_GRAMMAR;
        IGrammar grammar = GrammarFactory.getGrammar(op, new Parallelizer(op, 1));
        dim = 10;
        model = new Model(op, dim, 40, op.grammarOp.grammarType);
    }

    @Test
    public void testCompose() {
        val child1Vec = Nd4j.ones(dim, 1);
        val child2Vec = Nd4j.ones(dim, 1);

        model.getParams().setW(Nd4j.ones(dim, 2 * dim));
        val modelParentVec = model.compose(child1Vec, child2Vec);

        // verify it is a column vector
        assertTrue(modelParentVec.isColumnVector());

        // verify the shape is (10, 1)
        int[] shape = modelParentVec.shape();
        assertEquals(shape[0], dim);
        assertEquals(shape[1], 1);

        INDArray trueParentVec = sigmoid(Nd4j.ones(dim, 1).muli(2 * dim));
        assertEquals(trueParentVec, modelParentVec);
    }

    @Test
    public void testComposeDerivative() {

    }

    @Test
    public void testEnergyWord() {
        val vec = Nd4j.rand(dim, 1);
        double modelEnergy = model.energyWord(vec);

        float trueEnergy = identity(
                model
                        .getParams()
                        .getU()
                        .transpose()
                        .mmul(vec))
                .getFloat(0, 0);

        assertEquals(String.format("True: %f,Model: %f", trueEnergy, modelEnergy), trueEnergy,
                modelEnergy, .0001);
    }

    @Test
    public void testEnergyWordDerivative() {

    }

    @Test
    public void testLinearComposition() {
        val child1Vec = Nd4j.ones(dim, dim);
        val child2Vec = Nd4j.ones(dim, dim);
        val parentVec = Nd4j.ones(dim, dim);

        model.getParams().setU(Nd4j.ones(dim, 1));
        model.getParams().setH1(Nd4j.ones(dim, 1));
        model.getParams().setH2(Nd4j.ones(dim, 1));

        INDArray linearCompVec = model.linearComposition(parentVec, child1Vec, child2Vec);

        INDArray trueVec = Nd4j.ones(dim, 1).muli(3 * dim);

        assertEquals(trueVec, linearCompVec);

    }

    @Test
    public void testEnergyComp() {
        val child1Vec = Nd4j.ones(dim, 1);
        val child2Vec = Nd4j.ones(dim, 1);
        val parentVec = Nd4j.ones(dim, 1);

        model.getParams().setU(Nd4j.ones(dim, 1));
        model.getParams().setH1(Nd4j.ones(dim, 1));
        model.getParams().setH2(Nd4j.ones(dim, 1));

        double modelEnergy = model.energyComp(parentVec, child1Vec, child2Vec);

        float trueEnergy = 3 * dim;

        assertEquals(String.format("True: %f,Model: %f", trueEnergy, modelEnergy), trueEnergy,
                modelEnergy, 0.0001);
    }

    @Test
    public void testEnergyCompDerivative() {

    }

    @Test
    @Ignore
    public void testword2vec() {
        // TODO:: Complete the test after writing Word class
        assertTrue(false);
    }

    @Test
    public void testLinearWord() {

    }

    @Test
    public void testCalculateZ() {
        model.getParams().setU(Nd4j.ones(dim, 1).muli(1.0/dim));
        model.getParams().setX(Nd4j.ones(40, dim));

        double Z = model.calculateZ();

        double trueZ = Math.exp(-1) * 40;

        assertEquals(trueZ, Z, 0.0001);
    }

    @Test
    public void testExpectedl() {

        model.getParams().setU(Nd4j.ones(dim, 1).mul(1.0/dim));
        model.getParams().setH1(Nd4j.ones(dim, 1).mul(1.0/dim));
        model.getParams().setH2(Nd4j.ones(dim, 1).mul(1.0/dim));

        int len = 6;
        INDArray[] compMatrices = new INDArray[len];
        INDArray[] Eij = new INDArray[len];
        INDArray[][] phraseMatrix = new INDArray[len][len + 1];
        for (int i = 1; i < len; i++) {
            compMatrices[i] = Nd4j.ones(dim, 1);
            phraseMatrix[0][i] = Nd4j.ones(dim, 1);
            phraseMatrix[i][len] = Nd4j.ones(dim, 1);
            Eij[i] = Nd4j.ones(dim, 1).muli(i);
        }

        INDArray expectedArr = model.Expectedl(0, 6, Eij,
                                        compMatrices, phraseMatrix,
                                        1, new int[]{dim, 1});

        INDArray trueArray = Nd4j.ones(dim, 1).muli(3);

        assertEquals(trueArray, expectedArr);


    }

    @Test
    public void testExpectedV() {
        model.getParams().setU(Nd4j.ones(dim, 1).mul(1.0/dim));
        model.getParams().setX(Nd4j.ones(40, dim));
        INDArray expectedArr =
            model.ExpectedV(new Function<INDArray, INDArray>() {
                @Nullable
                public INDArray apply(@Nullable INDArray indArray) {
                    return identity(indArray);
                }
            }, new int[]{dim, 1});

        INDArray trueArray = Nd4j.ones(dim, 1);

        assertEquals(trueArray, expectedArr);


    }

    @Test(expected = RuntimeException.class)
    @Ignore
    public void testword2vecOutOfRangeException() {
        // TODO:: Complete the test after writing Word class
        assertTrue(false);
    }

    @Test(expected = RuntimeException.class)
    public void testcomposeChild1SizeException() {
        val vec10d = Nd4j.rand(dim, 1);
        val vec12d = Nd4j.rand(dim + 2, 1);
        model.compose(vec12d, vec10d);
    }

    @Test(expected = RuntimeException.class)
    public void testcomposeChild2SizeException() {
        val vec10d = Nd4j.rand(dim, 1);
        val vec12d = Nd4j.rand(dim + 2, 1);
        model.compose(vec10d, vec12d);
    }


    @Test(expected = RuntimeException.class)
    public void testcomposeChild1ShapeException() {
        val mat2d = Nd4j.rand(dim, 2);
        val vec = Nd4j.rand(dim, 1);
        model.compose(mat2d, vec);
    }

    @Test(expected = RuntimeException.class)
    public void testcomposeChild2ShapeException() {
        val mat2d = Nd4j.rand(dim, 2);
        val vec = Nd4j.rand(dim, 1);
        model.compose(vec, mat2d);
    }

    @Test(expected = RuntimeException.class)
    public void testenergySizeException() {
        val vec12d = Nd4j.rand(dim + 2, 1);
        model.unProbabilityWord(vec12d);
    }

    @Test(expected = RuntimeException.class)
    @Ignore
    public void testenergyShapeException() {
        val mat2d = Nd4j.rand(dim, 2);
        model.unProbabilityWord(mat2d);
    }



}
