package com.kushalarora.compositionalLM.lang;

import java.io.Serializable;
import java.util.List;

/**
 * Created by karora on 6/24/15.
 */
public interface IInsideOutsideScore extends Serializable {
    /**
     *
     * Returns \pi(i,j,k), an array of shape (n, n, n)
     * where n is length of the sentence. This array reperesents
     * sum of probability of all trees spanning (i,j) split at k
     * @return insideSpanSplitProb array
     */
    public double[][][] getInsideSpanSplitProb();

    /***
     * Returns \pi(i,j), an array of shape(n,n)
     * where n is length of sentence. This array represents
     * summ of probability of all trees spanning (i,j)
     * @return  insideSpanProb array
     */
    public double[][] getInsideSpanProb();

    public double[][][] getOutsideSpanWParentScore();

    public double[][][] getMuScore();


    public double[][][][] getMuSpanSplitScoreWParent();



    /**
     * Return the sentence currently being processed
     * @return sentence
     */
    public List<Word> getCurrentSentence();

    public void computeInsideOutsideProb();

}
