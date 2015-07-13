package com.kushalarora.compositionalLM.options;

import java.io.Serializable;

/**
 * Created by karora on 6/22/15.
 */
public class Options implements Serializable {
    public static enum FILE_TYPE {TEXT, SERIALIZED}
    public boolean train;
    public boolean parse;
    public boolean nbestRescore;
    public boolean verbose;
    public GrammarOptions grammarOp = new GrammarOptions();
    public TrainOptions trainOp = new TrainOptions();
    public TestOptions testOp = new TestOptions();
    public ModelOptions modelOp = new ModelOptions();
}
