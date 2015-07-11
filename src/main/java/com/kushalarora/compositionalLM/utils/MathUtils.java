package com.kushalarora.compositionalLM.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by karora on 6/18/15.
 */
public class MathUtils {
    /**
     * Returns an indicator column vector with zeros at all but <code>index</code> position
     * @param index Index for indicator vector
     * @param dimension Length of the vector to be created
     * @return an indicator vector for <code>index</code>  of size <code>dimension</code>
     */
    public static INDArray indicatorVector(int index, int dimension) {
        INDArray indVector = Nd4j.zeros(1, dimension);
        indVector.putScalar(index, 1.0f);
        return indVector;
    }
}
