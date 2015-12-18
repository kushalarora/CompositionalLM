package com.kushalarora.compositionalLM.lang;

import edu.stanford.nlp.parser.lexparser.Lexicon;
import lombok.Getter;
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

    @Getter
    protected Sentence sentence;
    @Getter
    protected int length;
    protected int numStates;

    public AbstractInsideOutsideScore(Sentence sentence, int numStates) {
        this.sentence = new Sentence(sentence.getIndex());
        this.sentence.addAll(sentence);
        this.sentence.add(new Word(Lexicon.BOUNDARY, length));
        length = this.sentence.size();
        this.numStates = numStates;

        // zero out some stuff first in case we recently
        // ran out of memory and are reallocating
        log.info("Starting array allocation");
/*            iScore = new double[length][length + 1][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    iScore[start][end] = new double[numStates];
                }
            }*/
        iScore = SparseMatrix.Factory.zeros(length, length + 1, numStates);

/*
            oScore = new double[length][length + 1][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    oScore[start][end] = new double[numStates];
                }
            }*/
        oScore = SparseMatrix.Factory.zeros(length, length + 1, numStates);

//            iSpanScore = new double[length][length + 1];
        iSpanScore = SparseMatrix.Factory.zeros(length, length + 1);

/*            iSpanSplitScore = new double[length][length + 1][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    // splits
                    iSpanSplitScore[start][end] = new double[length];
                }
            }*/
        iSpanSplitScore = SparseMatrix.Factory.zeros(length, length + 1, length);

/*            oSpanWParentScore = new double[length][length + 1][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    // parents
                    oSpanWParentScore[start][end] = new double[length + 1];
                }
            }*/
        oSpanWParentScore = SparseMatrix.Factory.zeros(length, length + 1, length + 1);

/*
            muScore = new double[length][length + 1][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    muScore[start][end] = new double[numStates];
                }
            }*/
        muScore = SparseMatrix.Factory.zeros(length, length + 1, numStates);

/*
            muSpanSplitScoreWParent = new double[length][length + 1][][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    // splits
                    muSpanSplitScoreWParent[start][end] = new double[length][];
                    for (int split = start; split < end; split++) {
                        // parents
                        muSpanSplitScoreWParent[start][end][split] = new double[length + 1];
                    }
                }
            }*/
        muSpanSplitScoreWParent = SparseMatrix.Factory.zeros(length, length + 1, length, length + 1);

        log.info("Finished allocating arrays of length {}", length);
    }

    public double getScore(SparseMatrix matrix, long... indexes) {
        return matrix.getAsDouble(indexes);
    }

    protected synchronized void setScore(SparseMatrix matrix, double value, long... indexes) {
        matrix.setAsDouble(value, indexes);
    }

    protected synchronized void addToScore(SparseMatrix matrix, double value, long... indexes) {
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

}
