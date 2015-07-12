package com.kushalarora.compositionalLM.lang.stanford;

import com.kushalarora.compositionalLM.lang.Word;
import edu.stanford.nlp.process.LexedTokenFactory;

/**
 * Created by karora on 7/11/15.
 */
public class WordTokenFactory implements LexedTokenFactory<Word> {
    public Word makeToken(String str, int begin, int length) {
        return new Word(str, -1);
    }

}
