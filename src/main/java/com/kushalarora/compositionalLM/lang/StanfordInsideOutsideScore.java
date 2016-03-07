package com.kushalarora.compositionalLM.lang;

import com.kushalarora.compositionalLM.optimizer.IIndexed;
import edu.stanford.nlp.parser.lexparser.Lexicon;
import org.ujmp.core.SparseMatrix;

/**
 * Created by arorak on 9/15/15.
 */
public class StanfordInsideOutsideScore extends AbstractInsideOutsideScore {

    protected transient SparseMatrix iSplitSpanStateScore;
    protected transient SparseMatrix oSpanStateScoreWParent;

    public StanfordInsideOutsideScore(Sentence sentence, int numStates) {
        super(sentence.size() + 1, numStates);
        this.sentence = new Sentence(sentence.getIndex());
        this.sentence.addAll(sentence);
        this.sentence.add(new Word(Lexicon.BOUNDARY, sentence.size()));
            /*
            oSpanStateScoreWParent = new double[length][length + 1][][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    // parents
                    oSpanStateScoreWParent[start][end] = new double[length + 1][];
                    for (int parent = 0; parent < start; parent++) {
                        // states
                        oSpanStateScoreWParent[start][end][parent] = new double[numStates];
                    }

                    for (int parent = end; parent <= length; parent++) {
                        // states
                        oSpanStateScoreWParent[start][end][parent] = new double[numStates];
                    }
                }
            }*/
        oSpanStateScoreWParent = SparseMatrix.Factory.zeros(length, length + 1, length + 1, numStates);


/*
            iSplitSpanStateScore = new double[length][length + 1][][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    iSplitSpanStateScore[start][end] = new double[length][];
                    for (int split = start; split < end; split++) {
                        // states
                        iSplitSpanStateScore[start][end][split] = new double[numStates];
                    }
                }
            }*/
        iSplitSpanStateScore = SparseMatrix.Factory.zeros(length, length + 1, length, numStates);
    }

    public void clearTempArrays() {
        iSplitSpanStateScore = null;
        oSpanStateScoreWParent = null;
    }

    public void clearNonSpanArrays() {
        iScore = null;
        oScore = null;
        muScore = null;
    }

    public IIndexed get(int index) {
        return sentence.get(index);
    }

    public int getIndex() {
        return sentence.getIndex();
    }

    public int getSize() {
        return sentence.getSize();
    }
}

