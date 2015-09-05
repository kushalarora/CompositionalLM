package com.kushalarora.compositionalLM.lang;

import edu.stanford.nlp.parser.lexparser.Lexicon;
import lombok.extern.slf4j.Slf4j;
import org.ujmp.core.SparseMatrix;

import java.util.List;

/**
 * Created by karora on 6/24/15.
 */
@Slf4j
public abstract class AbstractInsideOutsideScore implements IInsideOutsideScore {
    // inside scores
    // start idx, end idx, state -> logProb (ragged; null for end <= start)
    protected transient SparseMatrix iScore;
    protected SparseMatrix iSpanScore;
    protected SparseMatrix iSpanSplitScore;

    // outside scores
    // start idx, end idx, state -> logProb
    protected transient SparseMatrix oScore;
    protected SparseMatrix oSpanWParentScore;

    protected transient SparseMatrix muScore;
    protected SparseMatrix muSpanSplitScoreWParent;

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

    public double getScore(SparseMatrix matrix, long... indexes) {
        return matrix.getAsDouble(indexes);
    }

    protected void setScore(SparseMatrix matrix, double value, long... indexes) {
        matrix.setAsDouble(value, indexes);
    }

    protected void addToScore(SparseMatrix matrix, double value, long... indexes) {
        setScore(matrix, value + getScore(matrix, indexes), indexes);
    }


    public SparseMatrix getInsideScores() {
        return iScore;
    }

    public SparseMatrix getOutsideScores() {
        return oScore;
    }

    public SparseMatrix getInsideSpanSplitProb() {
        return iSpanSplitScore;
    }

    public SparseMatrix getInsideSpanProb() {
        return iSpanScore;
    }

    public SparseMatrix getOutsideSpanWParentScore() {
        return oSpanWParentScore;
    }

    public List<Word> getCurrentSentence() {
        return sentence;
    }

    public SparseMatrix getMuScore() {
        return muScore;
    }


    public SparseMatrix getMuSpanSplitScoreWParent() { return muSpanSplitScoreWParent; }

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
