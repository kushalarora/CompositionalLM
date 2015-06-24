package com.kushalarora.compositionalLM.lang;

import edu.berkeley.nlp.PCFGLA.CoarseToFineMaxRuleParser;
import edu.berkeley.nlp.PCFGLA.Grammar;
import edu.berkeley.nlp.PCFGLA.Lexicon;

import java.util.List;

/**
 * Created by karora on 6/20/15.
 */
public class BerkeleyCFGrammar extends CoarseToFineMaxRuleParser implements IGrammar {

    public BerkeleyCFGrammar(Grammar gr, Lexicon lex, double unaryPenalty, int endL, boolean viterbi, boolean sub, boolean score, boolean accurate, boolean variational, boolean useGoldPOS, boolean initializeCascade) {
        // TODO:: Make a lot of these defaults
        super(gr, lex, unaryPenalty,
                endL, viterbi, sub,
                score, accurate, variational,
                useGoldPOS, initializeCascade);
    }

    public void computeInsideOutsideProb(List<Word> sentence) {

    }

    public float[][][] getInsideSpanSplitProb() {
        return null;
    }

    public float[][] getOutsideSpanProb() {
        return null;
    }

    public List<Word> getCurrentSentence() {
        return null;
    }

}
