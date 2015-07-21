package com.kushalarora.compositionalLM.lang;

import edu.stanford.nlp.ling.HasWord;
import lombok.EqualsAndHashCode;

/**
 * Created by karora on 6/18/15.
 */

@EqualsAndHashCode
public class Word implements HasWord, edu.berkeley.nlp.io.HasWord {
    String word;
    int index;
    String signature;

    public Word(String word, int index) {
        this(word, index, word);
    }

    public Word(String word, int index, String signature) {
        this.word = word;
        this.index = index;
        this.signature = signature;
    }

    public int getIndex() {
        return index;
    }

    public String word() {
        return word;
    }

    public String getSignature() {
        return signature;
    }

    public void setWord(String word) {
        this.word = word;
    }

    @Override
    public String toString() {
        return String.format("%s(%d)", word, index);
    }
}
