package com.kushalarora.test.languagemodel;

import com.kushalarora.compositionalLM.languagemodel.CompositionalLM;
import com.kushalarora.compositionalLM.model.Parameters;
import edu.stanford.nlp.io.IOUtils;
import lombok.SneakyThrows;
import lombok.val;
import org.apache.commons.io.FileUtils;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;

import java.io.ObjectOutputStream;

import static junit.framework.Assert.assertEquals;
import static junit.framework.TestCase.assertTrue;

/**
 * Created by karora on 6/18/15.
 */
public class CompositionLMTest {

    public static CompositionalLM compositionalLM;
    public static Parameters trueParameters;
    public static String TRUE_FILENAME;

    @BeforeClass
    @SneakyThrows
    public static void setUpClass() {
        TRUE_FILENAME = FileUtils.getFile("src/test/resources/model.gz")
                .getAbsolutePath();

        ObjectOutputStream out = IOUtils.writeStreamFromString(TRUE_FILENAME);

        trueParameters = new Parameters(10, 100);
        out.writeObject(trueParameters);
        out.close();
        compositionalLM = new CompositionalLM(trueParameters);
    }

    @Test
    @SneakyThrows
    public void testSaveModelSerialized() {
        compositionalLM.saveModelSerialized("/tmp/model.gz");

        Parameters openedModel = compositionalLM.loadModelSerialized("/tmp/model.gz");
        assertEquals(trueParameters, openedModel);
    }

    @Test
    @Ignore
    public void testSaveModelText() {
        assertTrue(false);
    }

    @Test
    // TODO:: Write test in a proper way
    public void testLoadModelSerialized() {
        Parameters model = compositionalLM.loadModelSerialized(TRUE_FILENAME);
        assertEquals(trueParameters, model);
    }

    @Test
    @Ignore
    public void testLoadModelText() {
        assertTrue(false);
    }
}
