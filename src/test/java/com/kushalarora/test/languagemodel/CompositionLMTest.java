package com.kushalarora.test.languagemodel;

import com.kushalarora.compositionalLM.languagemodel.CompositionalLM;
import com.kushalarora.compositionalLM.model.Model;
import edu.stanford.nlp.io.IOUtils;
import lombok.SneakyThrows;
import org.apache.commons.io.FileUtils;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;

import java.io.ObjectOutputStream;

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
        TRUE_FILENAME = FileUtils.getFile("src/test/resources/model.gz")
                .getAbsolutePath();

        ObjectOutputStream out = IOUtils.writeStreamFromString(TRUE_FILENAME);

        trueModel = new Model(10, 100);
        out.writeObject(trueModel);
        out.close();
        compositionalLM = new CompositionalLM(trueModel);
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
}
