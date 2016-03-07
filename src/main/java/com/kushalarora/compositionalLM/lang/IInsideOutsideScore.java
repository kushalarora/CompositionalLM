package com.kushalarora.compositionalLM.lang;

import com.kushalarora.compositionalLM.optimizer.IIndexedSized;
import org.ujmp.core.SparseMatrix;

import java.io.Serializable;
import java.util.List;

/**
 * Created by karora on 6/24/15.
 */
public interface IInsideOutsideScore extends Serializable, IIndexedSized {
    /**
     *
     * Returns \pi(i,j,k), an array of shape (n, n, n)
     * where n is length of the sentence. This array reperesents
     * sum of probability of all trees spanning (i,j) split at k
     * @return insideSpanSplitProb array
     */
    public SparseMatrix getInsideSpanSplitProb();

    /***
     * Returns \pi(i,j), an array of shape(n,n)
     * where n is length of sentence. This array represents
     * summ of probability of all trees spanning (i,j)
     * @return  insideSpanProb array
     */
    public SparseMatrix getInsideSpanProb();

    public SparseMatrix getOutsideSpanWParentScore();

    public SparseMatrix getMuScore();


    public SparseMatrix getMuSpanSplitScoreWParent();


    /**
     * Return the sentence currently being processed
     * @return sentence
     */
    public List<Word> getCurrentSentence();

    public double getScore(SparseMatrix matrix, long... indexes);

}
