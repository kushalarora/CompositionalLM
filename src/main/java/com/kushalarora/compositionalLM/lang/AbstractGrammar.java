package com.kushalarora.compositionalLM.lang;

/**
 * Created by karora on 7/19/15.
 */
public abstract class AbstractGrammar implements IGrammar {

    public abstract AbstractInsideOutsideScore getScore(Sentence sentence);

    public AbstractInsideOutsideScore computeScore(Sentence sentence) {
        AbstractInsideOutsideScore score = getScore(sentence);
        return score;
    }


    /**
     * Intialize charts with lexicons.
     * Implemented by the derived class using
     * the specific grammar
     */
    public abstract void doLexScores(AbstractInsideOutsideScore s);

    /**
     * Compute the inside scores, insideSpanSplit,
     * insideSpan scores.
     */
    public abstract void doInsideScores(AbstractInsideOutsideScore s);

    /**
     * Compute outside, outside span and
     * outside span with parent scores
     */
    public abstract void doOutsideScores(AbstractInsideOutsideScore s);


    public abstract void doMuScore(AbstractInsideOutsideScore s);


}
