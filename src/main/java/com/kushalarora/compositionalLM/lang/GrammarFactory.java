package com.kushalarora.compositionalLM.lang;

import com.kushalarora.compositionalLM.lang.stanford.StanfordGrammar;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.options.Options;
import edu.berkeley.nlp.PCFGLA.Grammar;
import edu.berkeley.nlp.PCFGLA.Lexicon;
import edu.berkeley.nlp.PCFGLA.ParserData;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import lombok.NonNull;
import lombok.val;

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
        BERKELEY_GRAMMAR("berkeley"),
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

    public static IGrammar getGrammar(Options op) {
        switch (op.grammarOp.grammarType) {
            case BERKELEY_GRAMMAR:
                @NonNull ParserData parserData = ParserData.Load(op.grammarOp.filename);
                @NonNull Grammar gr = parserData.getGrammar();
                @NonNull Lexicon lexicon = parserData.getLexicon();
                // return berkeley parser
                return null;
            case STANFORD_GRAMMAR:
                // TODO: Do similar thing for stanford grammar
                LexicalizedParser model = LexicalizedParser.loadModel(op.grammarOp.filename);
                return new StanfordGrammar(op, model);
            default:
                throw new RuntimeException("Invalid Grammar Type: " + op.grammarOp.grammarType);
        }
    }
}
