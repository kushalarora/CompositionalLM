package com.kushalarora.compositionalLM.options;

import lombok.ToString;
import org.apache.commons.configuration.Configuration;

import java.io.Serializable;

/**
 * Created by karora on 7/11/15.
 */
@ToString
public class ModelOptions implements Serializable {
    public Options.FileType outType;
    public String outFilename;
    public Options.FileType inType;
    public String inFilename;
    public int dimensions;


    public ModelOptions(Configuration config) {
        outType =
                Options.FileType.fromString(
                config.getString("outType", "serialized"));

        outFilename =
                config.getString("outFile", "/tmp/model.ser.gz");

        inType  =
                Options.FileType.fromString(
                        config.getString("inType", "serialized"));

        inFilename =
                config.getString("inFile", null);

        dimensions =
                config.getInt("dimensions", 10);
    }
}
