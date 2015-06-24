package com.kushalarora.compositionalLM.lang;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

/**
 * Created by karora on 6/20/15.
 */
public interface IGrammar {

    /**
     *
     * @param sentence
     */
    public void  computeInsideOutsideProb(List<Word> sentence);
    /**
     *
     * Returns \pi(i,j,k), an array of shape (n, n, n)
     * where n is length of the sentence. This array reperesents
     * sum of probability of all trees spanning (i,j) split at k
     * @return insideSpanSplitProb array
     */
    public float[][][] getInsideSpanSplitProb();

    /***
     * Returns \pi(i,j), an array of shape(n,n)
     * where n is length of sentence. This array represents
     * summ of probability of all trees spanning (i,j)
     * @return  insideSpanProb array
     */
    public float[][] getInsideSpanProb();

    /**
     *  Returns \beta(i,j) an array of shape(n,n) where
     *  n is the length of the sentence. This represents sum of
     *  all possible trees where span (i,j) is not expanded
     *  @return outsideSpanProb array
     */
    public float[][] getOutsideSpanProb();

    /**
     * Return the sentence currently being processed
     * @return sentence
     */
    public List<Word> getCurrentSentence();
}
