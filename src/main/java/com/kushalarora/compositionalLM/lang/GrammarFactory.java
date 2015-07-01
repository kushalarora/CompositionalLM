package com.kushalarora.compositionalLM.lang;

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
    public static enum GrammarType {
        BERKELEY_GRAMMAR,
        STANFORD_GRAMMAR
    }

    public static IGrammar getGrammar(GrammarType type, String filename, Options op) {
        switch (type) {
            case BERKELEY_GRAMMAR:
                @NonNull ParserData parserData = ParserData.Load(filename);
                @NonNull Grammar gr = parserData.getGrammar();
                @NonNull Lexicon lexicon = parserData.getLexicon();
                // return berkeley parser
                return null;
            case STANFORD_GRAMMAR:
                // TODO: Do similar thing for stanford grammar
                @NonNull val model = LexicalizedParser.loadModel(filename);
                return new StanfordGrammar(model.bg, model.ug, model.lex, op,
                        model.getOp(), model.stateIndex, model.wordIndex,
                        model.tagIndex);
            default:
                throw new RuntimeException("Invalid Grammar Type: " + type);
        }
    }
}
