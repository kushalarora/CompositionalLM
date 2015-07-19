package com.kushalarora.compositionalLM.lang;

import java.util.List;

/**
 * Created by karora on 7/19/15.
 */
public abstract class AbstractGrammar implements IGrammar {

    protected abstract AbstractInsideOutsideScorer getScore(List<Word> sentence);

    public AbstractInsideOutsideScorer computeScore(List<Word> sentence) {
        AbstractInsideOutsideScorer score = getScore(sentence);
        score.computeInsideOutsideProb();
        return score;
    }

}
