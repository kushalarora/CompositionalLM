package com.kushalarora.compositionalLM.options;

import com.kushalarora.compositionalLM.lang.GrammarFactory;
import com.kushalarora.compositionalLM.lang.TokenizerFactory;

/**
 * Created by karora on 6/22/15.
 */
public class GrammarOptions {
    public static int DEFAULT_MAX_LENGTH = 40;

    public static int maxLength = DEFAULT_MAX_LENGTH;

    public static GrammarFactory.GrammarType grammarType;

    public static String filename;

    public static TokenizerFactory.TokenizerType tokenizerType;
}
