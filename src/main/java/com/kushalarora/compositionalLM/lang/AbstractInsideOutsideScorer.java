package com.kushalarora.compositionalLM.lang;

import edu.stanford.nlp.parser.lexparser.Lexicon;
import lombok.extern.slf4j.Slf4j;

import java.util.List;

/**
 * Created by karora on 6/24/15.
 */
@Slf4j
public abstract class AbstractInsideOutsideScorer implements IInsideOutsideScorer {
    protected double[][][] iSpanSplitScore;
    protected double[][] iSpanScore;
    protected double[][][] oSpanWParentScore;
    protected double[][] oSpanScore;
    protected double[][][] muSpanSplitScore;
    protected double[][][] muScore;
    protected double[][][][] muSpanScoreWParent;
    // inside scores
    // start idx, end idx, state -> logProb (ragged; null for end <= start)
    protected double[][][] iScore;
    // outside scores
    // start idx, end idx, state -> logProb
    protected double[][][] oScore;

    protected List<Word> sentence;
    protected int length;
    protected int arraySize = 0;
    protected int myMaxLength = Integer.MAX_VALUE;

    public double[][][] getInsideScores() {
        return iScore;
    }

    public double[][][] getOutsideScores() {
        return oScore;
    }

    public double[][][] getInsideSpanSplitProb() {
        return iSpanSplitScore;
    }

    public double[][] getInsideSpanProb() {
        return iSpanScore;
    }

    public double[][][] getOutsideSpanWParentScore() {
        return oSpanWParentScore;
    }

    public double[][] getOutsideSpanProb() {
        return oSpanScore;
    }

    public List<Word> getCurrentSentence() {
        return sentence;
    }

    public double[][][] getMuScore() {
        return muScore;
    }

    public double[][][] getMuSpanSplitScore() {
        return muSpanSplitScore;
    }

    public double[][][][] getMuSpanScoreWParent() { return muSpanScoreWParent; }

    public abstract void clearArrays();

    public abstract void considerCreatingArrays(int length);

    public abstract void initializeScoreArrays(int length);

    /**
     * Intialize charts with lexicons.
     * Implemented by the derived class using
     * the specific grammar
     */
    public abstract void doLexScores(List<Word> sentence);

    /**
     * Compute the inside scores, insideSpanSplit,
     * insideSpan scores.
     */
    public abstract void doInsideScores(List<Word> sentence);

    /**
     * Compute outside, outside span and
     * outside span with parent scores
     */
    public abstract void doOutsideScores(List<Word> sentence);


    public abstract void doMuScore(List<Word> sentence);
}
