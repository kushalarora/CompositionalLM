package com.kushalarora.compositionalLM.lang;

import java.util.List;

/**
 * Created by karora on 7/19/15.
 */
public abstract class AbstractGrammar implements IGrammar {

    public abstract AbstractInsideOutsideScore getScore(List<Word> sentence);

    public AbstractInsideOutsideScore computeScore(List<Word> sentence) {
        AbstractInsideOutsideScore score = getScore(sentence);
        score.computeInsideOutsideProb();
        return score;
    }

}
