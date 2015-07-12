package com.kushalarora.compositionalLM.lang;

import edu.stanford.nlp.process.Tokenizer;

import javax.el.MethodNotFoundException;
import java.util.List;

/**
 * Created by karora on 7/12/15.
 */
public abstract class TokenizerWrapper implements Tokenizer<Word> {

    public abstract boolean hasNext();

    public abstract Word next();

    public void remove() {
        throw new MethodNotFoundException("Derived from Stanford Tokenizer");
    }

    public Word peek() {
        return null;
    }

    public List<Word> tokenize() {
        return null;
    }
}
