package com.kushalarora.compositionalLM.lang;

import com.kushalarora.compositionalLM.lang.stanford.WordTokenFactory;
import com.kushalarora.compositionalLM.options.Options;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.Tokenizer;
import edu.stanford.nlp.process.WhitespaceTokenizer;

import java.io.Reader;
import java.util.Iterator;

/**
 * Created by karora on 7/12/15.
 */
public class TokenizerFactory implements edu.stanford.nlp.process.TokenizerFactory<Word> {

    private final IGrammar grammar;
    Options op;

    public TokenizerFactory(Options op, IGrammar grammar) {
        this.op = op;
        this.grammar = grammar;
    }

    public Iterator<Word> getIterator(Reader r) {
        return null;
    }

    public enum TokenizerType {
        STANFORD_PTB_TOKENIZER("StanfordPTB"),
        STANFORD_WORD_TOKENIZER("StanfordWord");

        private String text;

        TokenizerType(String text) {
            this.text = text;
        }

        public String getText() {
            return this.text;
        }

        public static TokenizerType fromString(String text) {
            if (text != null) {
                for (TokenizerType b : TokenizerType.values()) {
                    if (text.equalsIgnoreCase(b.text)) {
                        return b;
                    }
                }
            }
            return null;
        }
    }

    public TokenizerWrapper getTokenizer(Reader reader, String extraOptions) {
        switch (op.grammarOp.tokenizerType) {
            case STANFORD_PTB_TOKENIZER:
                final PTBTokenizer ptbTokenizer =
                        new PTBTokenizer<Word>(
                                reader, new WordTokenFactory(grammar, op), extraOptions);

                return new TokenizerWrapper() {
                    @Override
                    public boolean hasNext() {
                        return ptbTokenizer.hasNext();
                    }

                    @Override
                    public Word next() {
                        return (Word) ptbTokenizer.next();
                    }
                };

            case STANFORD_WORD_TOKENIZER:
                final WhitespaceTokenizer.WhitespaceTokenizerFactory factory =
                        new WhitespaceTokenizer.WhitespaceTokenizerFactory(
                                new WordTokenFactory(grammar, op), extraOptions);

                final Tokenizer tokenizer = factory.getTokenizer(reader);

                return new TokenizerWrapper() {
                    @Override
                    public boolean hasNext() {
                        return tokenizer.hasNext();
                    }

                    @Override
                    public Word next() {
                        return (Word) tokenizer.next();
                    }
                };

            default:
                throw new RuntimeException("Invalid Tokenizer Type: " + op.grammarOp.tokenizerType);
        }
    }

    public Tokenizer<Word> getTokenizer(Reader r) {
        return getTokenizer(r, "");
    }

    public void setOptions(String options) {

    }
}
