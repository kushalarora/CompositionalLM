package com.kushalarora.compositionalLM.options;

import com.kushalarora.compositionalLM.lang.GrammarFactory;
import com.kushalarora.compositionalLM.lang.TokenizerFactory;

import java.io.Serializable;

/**
 * Created by karora on 6/22/15.
 */
public class GrammarOptions implements Serializable {
    public static int DEFAULT_MAX_LENGTH = 40;

    public static int maxLength = DEFAULT_MAX_LENGTH;

    public static GrammarFactory.GrammarType grammarType =
            GrammarFactory.GrammarType.STANFORD_GRAMMAR;

    public static String filename = "src/resources/englishPCFG.ser.gz";

    public static TokenizerFactory.TokenizerType tokenizerType =
            TokenizerFactory.TokenizerType.STANFORD_PTB_TOKENIZER;

    public static boolean lowerCase = false;
}
