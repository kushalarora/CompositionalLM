package com.kushalarora.compositionalLM.options;

import java.io.Serializable;

/**
 * Created by karora on 6/22/15.
 */
public class TestOptions implements Serializable {
    public boolean lengthNormalization;
    public boolean nbestRescore;
    public boolean parse;
    public String[] nbestFiles;
    public String[] parseFiles;
}
