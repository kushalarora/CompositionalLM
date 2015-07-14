package com.kushalarora.compositionalLM.options;

import lombok.ToString;

import java.io.Serializable;

/**
 * Created by karora on 6/22/15.
 */
@ToString
public class Options implements Serializable {
    public static enum FILE_TYPE {TEXT, SERIALIZED}

    public boolean train = false;
    public boolean parse = false;
    public boolean nbestRescore = false;
    public boolean verbose = false;
    public GrammarOptions grammarOp = new GrammarOptions();
    public TrainOptions trainOp = new TrainOptions();
    public TestOptions testOp = new TestOptions();
    public ModelOptions modelOp = new ModelOptions();

}
