package com.kushalarora.compositionalLM.options;

import lombok.ToString;
import org.apache.commons.configuration.Configuration;

import java.io.Serializable;

/**
 * Created by karora on 6/22/15.
 */
@ToString
public class TestOptions implements Serializable {
    public boolean nbestRescore;
    public boolean parse;
    public String[] nbestFiles;
    public String[] parseFiles;
    public String[] testFiles;
    public String visualizationFile;
    public String outputFile;
    public int testBatchSize;

    public TestOptions(Configuration config) {
        nbestRescore =
                config.getBoolean("nBestRescore", false);

        parse =
                config.getBoolean("parse", false);

        nbestFiles =
                config.getStringArray("nBestFiles");

        parseFiles  =
                config.getStringArray("parseFiles");

        visualizationFile =
                config.getString("visualizationFile");

        testFiles  =
                config.getStringArray("testFiles");

        outputFile =
                config.getString("outputFile", "./output.txt");

        testBatchSize =
                config.getInt("testBatchSize", 100);
    }
}
