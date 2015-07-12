package com.kushalarora.compositionalLM.lang;

import java.util.List;

/**
 * Grammar interface to provide inside and outside scores
 * that act as a prior for compositional grammar.
 * Created by karora on 6/20/15.
 */
public interface IGrammar {

    /**
     * Use the grammar to compute the inside and outside probability
     * and return the IInsideOutsideScores object.
     *
     * @param sentence sentence to process
     */
    public IInsideOutsideScores computeInsideOutsideProb(List<Word> sentence);

    public List<Word> getVocab();

    public int getVocabSize();
}