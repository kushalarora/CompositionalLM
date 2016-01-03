package com.kushalarora.compositionalLM.options;

import com.kushalarora.compositionalLM.lang.GrammarFactory;
import com.kushalarora.compositionalLM.lang.TokenizerFactory;
import lombok.ToString;
import org.apache.commons.configuration.Configuration;

import java.io.Serializable;

/**
 * Created by karora on 6/22/15.
 */
@ToString
public class GrammarOptions implements Serializable {
    public static int DEFAULT_MAX_LENGTH = 50;

    public int maxLength = DEFAULT_MAX_LENGTH;

    public GrammarFactory.GrammarType grammarType;

    public String filename;

    public TokenizerFactory.TokenizerType tokenizerType;

    public boolean lowerCase;
    public boolean newLineDelimiter;


    public GrammarOptions(Configuration config) {
        maxLength = config.getInt("maxLength",
                DEFAULT_MAX_LENGTH);

        grammarType =
                GrammarFactory.GrammarType.fromString(
                        config.getString("grammarType",
                                "stanford"));

        filename = config.getString("grammarFile",
                "src/resources/englishPCFG.ser.gz");

        tokenizerType =
                TokenizerFactory.TokenizerType.fromString(
                        config.getString("tokenizerType",
                                "stanfordPTB"));

        lowerCase = config.getBoolean("lowerCase", false);

        newLineDelimiter = config.getBoolean("nlDelim", false);


    }
}
