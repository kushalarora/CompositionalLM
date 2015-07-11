package com.kushalarora.compositionalLM.lang;

import edu.stanford.nlp.ling.HasWord;

/**
 * Created by karora on 6/18/15.
 */
public class Word implements HasWord, edu.berkeley.nlp.io.HasWord{
    String word;
    int index;

    public Word(String word, int index) {
        this.word = word;
        this.index = index;
    }

    public int getIndex() {
        // TODO:: Implement word class
        return index;
    }


    public String word() {
        return word;
    }

    public void setWord(String word) {
        this.word = word;
    }
}
