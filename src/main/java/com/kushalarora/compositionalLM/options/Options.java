package com.kushalarora.compositionalLM.options;

/**
 * Created by karora on 6/22/15.
 */
public class Options {
    public static enum FILE_TYPE {TEXT, SERIALIZED}
    public boolean train;
    public boolean parse;
    public boolean nbestRescore;
    public boolean verbose;
    public GrammarOptions grammarOp;
    public TrainOptions trainOp;
    public TestOptions testOp;
    public ModelOptions modelOp;
}
