package com.kushalarora.compositionalLM.options;

/**
 * Created by karora on 6/14/15.
 */
public class ParserOptions {
    public enum FILE_TYPE {TEXT, SERIALIZED}
    public String[] trainFiles = null;
    public boolean train = false;

    public boolean nbestRescore = false;
    public String[] nbestFiles = null;

    public String[] validationFiles = null;

    public boolean parse = false;
    public String[] parseFiles = null;

    public FILE_TYPE grammarTypeFile = null;
    public String grammarFilePath = null;

    public String outputFilePath = null;
    public FILE_TYPE outputTypeFile = null;

    public String modelFilePath = null;
    public FILE_TYPE modelTypeFile = null;

}
