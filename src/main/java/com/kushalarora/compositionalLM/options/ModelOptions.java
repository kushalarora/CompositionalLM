package com.kushalarora.compositionalLM.options;

import lombok.ToString;

import java.io.Serializable;

/**
 * Created by karora on 7/11/15.
 */
@ToString
public class ModelOptions implements Serializable {
    public Options.FILE_TYPE outType = Options.FILE_TYPE.SERIALIZED;
    public String outFilename = null;
    public Options.FILE_TYPE inType = Options.FILE_TYPE.SERIALIZED;
    public String inFilename = null;
    public int dimensions = 10;
}
