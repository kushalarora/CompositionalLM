package com.kushalarora.compositionalLM.options;

import lombok.SneakyThrows;
import lombok.ToString;
import org.apache.commons.configuration.ConfigurationException;
import org.apache.commons.configuration.PropertiesConfiguration;
import org.apache.commons.io.FileUtils;

import java.io.Serializable;

/**
 * Created by karora on 6/22/15.
 */
@ToString
public class Options implements Serializable {


    public enum FileType {
        TEXT("text"),
        SERIALIZED("serialized");

        private String text;

        FileType(String text) {
            this.text = text;
        }

        public String getText() {
            return this.text;
        }

        public static FileType fromString(String text) {
            if (text != null) {
                for (FileType b : FileType.values()) {
                    if (text.equalsIgnoreCase(b.text)) {
                        return b;
                    }
                }
            }
            return null;
        }
    }

    public boolean visualize;
    public boolean train;
    public boolean parse;
    public boolean test;
    public boolean nbestRescore;
    public boolean calcCE;
    public boolean debug;
    public GrammarOptions grammarOp;
    public TrainOptions trainOp;
    public TestOptions testOp;
    public ModelOptions modelOp;
    public InputOptions inputOp;
    public CacheOptions cacheOp;

    @SneakyThrows
    public Options() throws ConfigurationException {
        train = false;
        parse = false;
        nbestRescore = false;
        visualize = false;
        debug = false;
        calcCE = false;

        PropertiesConfiguration config =
                new PropertiesConfiguration();

        config.load(
                FileUtils.getFile("configs/grammar.config")
                        .getAbsolutePath());
        grammarOp = new GrammarOptions(config);

        config.load(
                FileUtils.getFile("configs/train.config")
                        .getAbsolutePath());
        trainOp = new TrainOptions(config);

        config.load(
                FileUtils.getFile("configs/test.config")
                        .getAbsolutePath());

        testOp = new TestOptions(config);

        config.load(
                FileUtils.getFile("configs/model.config")
                        .getAbsolutePath());
        modelOp = new ModelOptions(config);

        config.load(
                FileUtils.getFile("configs/input.config")
                        .getAbsolutePath());
        inputOp = new InputOptions(config);


        config.load(
                FileUtils.getFile("configs/cache.config")
                        .getAbsolutePath());
        cacheOp = new CacheOptions(config);
    }
}
