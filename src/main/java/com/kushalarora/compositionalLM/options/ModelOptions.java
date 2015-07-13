package com.kushalarora.compositionalLM.options;

import java.io.Serializable;

/**
 * Created by karora on 7/11/15.
 */
public class ModelOptions implements Serializable {
    public static Options.FILE_TYPE outType;
    public static String outFilename;
    public static Options.FILE_TYPE inType;
    public static String inFilename;
    public static int dimensions = 10;
}
