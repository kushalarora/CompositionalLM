package com.kushalarora.compositionalLM.lang;

import edu.stanford.nlp.ling.HasWord;

/**
 * Created by karora on 6/18/15.
 */
public class Word implements HasWord, edu.berkeley.nlp.io.HasWord{
    String word;

    public Word(String word) {
        this.word = word;
    }

    public int getIndex() {
        // TODO:: Implement word class
        return 0;
    }


    public String word() {
        return word;
    }

    public void setWord(String word) {
        this.word = word;
    }
}
