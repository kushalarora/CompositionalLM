package com.kushalarora.compositionalLM.lang;

import com.kushalarora.compositionalLM.options.Options;
import edu.stanford.nlp.process.LexedTokenFactory;

/**
 * Created by karora on 7/11/15.
 */
public class StanfordWordTokenFactory implements LexedTokenFactory<Word> {

    private final IGrammar grammar;
    private final Options op;
    // TODO:: Remove Options here
    public StanfordWordTokenFactory(IGrammar grammar, Options op) {
        this.grammar = grammar;
        this.op = op;

    }

    public Word makeToken(String str, int begin, int length) {
        return grammar.getToken(str, begin);
    }
}
