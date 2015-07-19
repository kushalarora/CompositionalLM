package com.kushalarora.compositionalLM.lang;

import java.io.Serializable;
import java.util.List;

/**
 * Grammar interface to provide inside and outside scores
 * that act as a prior for compositional grammar.
 * Created by karora on 6/20/15.
 */
public interface IGrammar extends Serializable {

    public IInsideOutsideScorer computeScore(List<Word> sentence);

    public int getVocabSize();

    public Word getToken(String str, int loc);
}