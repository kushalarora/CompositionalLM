package com.kushalarora.compositionalLM.lang.stanford;

import com.kushalarora.compositionalLM.lang.IGrammar;
import com.kushalarora.compositionalLM.lang.Word;
import com.kushalarora.compositionalLM.options.Options;
import edu.stanford.nlp.process.LexedTokenFactory;

/**
 * Created by karora on 7/11/15.
 */
public class WordTokenFactory implements LexedTokenFactory<Word> {

    private final IGrammar grammar;
    private final Options op;
    // TODO:: Remove Options here
    public WordTokenFactory(IGrammar grammar, Options op) {
        this.grammar = grammar;
        this.op = op;

    }

    public Word makeToken(String str, int begin, int length) {
        return grammar.getToken(str, begin);
    }
}
