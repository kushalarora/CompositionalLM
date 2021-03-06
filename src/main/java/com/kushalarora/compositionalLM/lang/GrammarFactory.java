package com.kushalarora.compositionalLM.lang;

import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.options.Options;
import com.kushalarora.compositionalLM.utils.Parallelizer;
import edu.berkeley.nlp.PCFGLA.Grammar;
import edu.berkeley.nlp.PCFGLA.Lexicon;
import edu.berkeley.nlp.PCFGLA.ParserData;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import lombok.NonNull;

/**
 * Created by karora on 6/20/15.
 */

//TODO: There are various grammars and then
// there are various other techniques like
// markovization and parent annotation etc
// that these parsers employ
// See how to handle all that shit

public class GrammarFactory {
    public enum GrammarType {
        STANFORD_GRAMMAR("stanford");

        private String text;

        GrammarType(String text) {
            this.text = text;
        }

        public String getText() {
            return this.text;
        }

        public static GrammarType fromString(String text) {
            if (text != null) {
                for (GrammarType b : GrammarType.values()) {
                    if (text.equalsIgnoreCase(b.text)) {
                        return b;
                    }
                }
            }
            return null;
        }
    }

    public static IGrammar getGrammar(Options op, Model model, Parallelizer parallelizer) {
        switch (op.grammarOp.grammarType) {
            case STANFORD_GRAMMAR:
                // TODO: Do similar thing for stanford grammar
                LexicalizedParser lexicalizedParser = LexicalizedParser.loadModel(op.grammarOp.filename);
                    return new StanfordCompositionalGrammar(op, lexicalizedParser, model, parallelizer);
            default:
                throw new RuntimeException("Invalid Grammar Type: " + op.grammarOp.grammarType);
        }
    }

    public static IGrammar getGrammar(Options op, Parallelizer parallelizer) {
        switch (op.grammarOp.grammarType) {
            case STANFORD_GRAMMAR:
                // TODO: Do similar thing for stanford grammar
                LexicalizedParser lexicalizedParser = LexicalizedParser.loadModel(op.grammarOp.filename);
                return new StanfordCompositionalGrammar(op, lexicalizedParser, parallelizer);
            default:
                throw new RuntimeException("Invalid Grammar Type: " + op.grammarOp.grammarType);
        }
    }
}
