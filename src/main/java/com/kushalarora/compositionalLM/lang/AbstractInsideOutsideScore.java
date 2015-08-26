package com.kushalarora.compositionalLM.lang;

import edu.stanford.nlp.parser.lexparser.Lexicon;
import lombok.extern.slf4j.Slf4j;

import java.util.List;

/**
 * Created by karora on 6/24/15.
 */
@Slf4j
public abstract class AbstractInsideOutsideScore implements IInsideOutsideScore {
    // inside scores
    // start idx, end idx, state -> logProb (ragged; null for end <= start)
    protected transient double[][][] iScore;
    protected double[][] iSpanScore;
    protected double[][][] iSpanSplitScore;

    // outside scores
    // start idx, end idx, state -> logProb
    protected transient double[][][] oScore;
    protected double[][][] oSpanWParentScore;

    protected transient double[][][] muScore;
    protected double[][][][] muSpanSplitScoreWParent;

    protected Sentence sentence;
    protected int length;
    protected int arraySize = 0;
    protected static int myMaxLength = Integer.MAX_VALUE;


    public AbstractInsideOutsideScore(Sentence sentence) {
        this.sentence = new Sentence(sentence.getIndex());
        this.sentence.addAll(sentence);
        this.sentence.add(new Word(Lexicon.BOUNDARY, length));
        length = this.sentence.size();
    }

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

    public List<Word> getCurrentSentence() {
        return sentence;
    }

    public double[][][] getMuScore() {
        return muScore;
    }


    public double[][][][] getMuSpanSplitScoreWParent() { return muSpanSplitScoreWParent; }

    public abstract void clearArrays();

    public abstract void considerCreatingArrays();

    public abstract void initializeScoreArrays();

    /**
     * Intialize charts with lexicons.
     * Implemented by the derived class using
     * the specific grammar
     */
    public abstract void doLexScores();

    /**
     * Compute the inside scores, insideSpanSplit,
     * insideSpan scores.
     */
    public abstract void doInsideScores();

    /**
     * Compute outside, outside span and
     * outside span with parent scores
     */
    public abstract void doOutsideScores();


    public abstract void doMuScore();

}
