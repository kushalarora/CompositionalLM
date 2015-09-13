package com.kushalarora.test.languagemodel;

import com.kushalarora.compositionalLM.lang.GrammarFactory;
import com.kushalarora.compositionalLM.lang.stanford.StanfordGrammar;
import com.kushalarora.compositionalLM.languagemodel.CompositionalLM;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.options.Options;
import edu.stanford.nlp.io.IOUtils;
import lombok.SneakyThrows;
import org.apache.commons.configuration.ConfigurationException;
import org.apache.commons.io.FileUtils;
import org.apache.log4j.PropertyConfigurator;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;

import java.io.IOException;
import java.io.ObjectOutputStream;

import static com.kushalarora.compositionalLM.lang.GrammarFactory.getGrammar;
import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertEquals;

/**
 * Created by karora on 6/18/15.
 */
public class CompositionLMTest {

    public static CompositionalLM compositionalLM;
    public static Model trueModel;
    public static String TRUE_FILENAME;

    @BeforeClass
    @SneakyThrows
    public static void setUpClass() {
        TRUE_FILENAME = FileUtils.getFile("src/resources/model.gz")
                .getAbsolutePath();
        String absoluteFilePath = FileUtils
                .getFile("src/resources/englishPCFG.ser.gz")
                .getAbsolutePath();

        ObjectOutputStream out = IOUtils.writeStreamFromString(TRUE_FILENAME);
        Options op = new Options();

        op.grammarOp.grammarType = GrammarFactory.GrammarType.STANFORD_GRAMMAR;
        op.grammarOp.filename =  absoluteFilePath;

        StanfordGrammar sg = (StanfordGrammar)getGrammar(op);
        trueModel = new Model(10, sg);
        out.writeObject(trueModel);
        out.close();

        compositionalLM = new CompositionalLM(trueModel, op);
        PropertyConfigurator.configure("log4j.properties");

    }

    @Test
    @SneakyThrows
    public void testSaveModelSerialized() {
        compositionalLM.saveModelSerialized("/tmp/model.gz");

        Model openedModel = compositionalLM.loadModelSerialized("/tmp/model.gz");
        assertEquals(trueModel, openedModel);
    }

    @Test
    @Ignore
    public void testSaveModelText() {
        assertTrue(false);
    }

    @Test
    // TODO:: Write test in a proper way
    public void testLoadModelSerialized() {
        Model model = compositionalLM.loadModelSerialized(TRUE_FILENAME);
        assertEquals(trueModel, model);
    }

    @Test
    @Ignore
    public void testLoadModelText() {
        assertTrue(false);
    }


    @Test
    @Ignore
    public void testMain() throws Exception {
        String[] args =
                new String[] {"-train", /*"src/resources/train10", */
                        "-validate", /* "src/resources/valid4",*/
                        "-grammarType", "stanford",
                        "-grammarFile", "src/resources/englishPCFG.ser.gz",
                        "-saveOutputModelSerialized", "/tmp/tmpmodel.ser.gz",
                        "-lowercase,",
                                "-validBatchSize", "10"};
        CompositionalLM.main(args);
    }

    @Test
    @Ignore
    public void testMainParallel() throws Exception {
        String[] args =
                new String[] {"-train", "src/resources/train10",
                        "-validate", "src/resources/valid4",
                        "-grammarType", "stanford",
                        "-grammarFile", "src/resources/englishPCFG.ser.gz",
                        "-saveOutputModelSerialized", "/tmp/tmpmodel.ser.gz",
                        "-lowercase",
                        "-parallel", "-nThreads 2"};
        CompositionalLM.main(args);
    }
}
