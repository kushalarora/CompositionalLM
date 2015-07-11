package com.kushalarora.compositionalLM.lang;

import edu.stanford.nlp.parser.lexparser.Lexicon;
import lombok.extern.slf4j.Slf4j;

import java.util.List;

/**
 * Created by karora on 6/24/15.
 */
@Slf4j
public abstract class AbstractInsideOutsideScores implements IInsideOutsideScores {
    protected float[][][] iSpanSplitScore;
    protected float[][] iSpanScore;
    protected float[][][] oSpanWParentScore;
    protected float[][] oSpanScore;
    protected float[][][] muSpanSplitScore;
    protected float[][][] muScore;
    protected float[][][][] muSpanScoreWParent;
    // inside scores
    // start idx, end idx, state -> logProb (ragged; null for end <= start)
    protected float[][][] iScore;
    // outside scores
    // start idx, end idx, state -> logProb
    protected float[][][] oScore;

    List<Word> sentence;
    protected int length;
    protected int arraySize = 0;
    protected int myMaxLength = Integer.MAX_VALUE;

    public AbstractInsideOutsideScores(List<Word> sentence) {
        this.sentence = sentence;

        // Adding boundary symbol to the sentence
        // as grammar needs it

        // TODO:: Figure out how to give index
        this.sentence.add(new Word(Lexicon.BOUNDARY, length));
        this.length = sentence.size();
    }

    public float[][][] getInsideScores() {
        return iScore;
    }

    public float[][][] getOutsideScores() {
        return oScore;
    }

    public float[][][] getInsideSpanSplitProb() {
        return iSpanSplitScore;
    }

    public float[][] getInsideSpanProb() {
        return iSpanScore;
    }

    public float[][][] getOutsideSpanWParentScore() {
        return oSpanWParentScore;
    }

    public float[][] getOutsideSpanProb() {
        return oSpanScore;
    }

    public List<Word> getCurrentSentence() {
        return sentence;
    }

    public float[][][] getMuScore() {
        return muScore;
    }

    public float[][][] getMuSpanSplitScore() {
        return muSpanSplitScore;
    }

    public float[][][][] getMuSpanScoreWParent() { return muSpanScoreWParent; }

    public abstract void clearArrays();

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


    public abstract void computeMuSpanScore();
}
