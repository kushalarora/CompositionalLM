package com.kushalarora.test.languagemodel;

import com.kushalarora.compositionalLM.languagemodel.CompositionalLM;
import com.kushalarora.compositionalLM.model.Parameters;
import lombok.SneakyThrows;
import lombok.val;
import org.apache.commons.io.FileUtils;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;

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
    public static void setUpClass() {
        TRUE_FILENAME = FileUtils.getFile("src/test/resources/model.gz")
                .getAbsolutePath();

        trueParameters = new Parameters(10, 100);
        compositionalLM = new CompositionalLM(trueParameters);
    }

    @Test
    @SneakyThrows
    public void testSaveModelSerialized() {
        compositionalLM.saveModelSerialized("/tmp/model.gz");

        val openedModel = compositionalLM.loadModelSerialized("/tmp/model.gz");
        assertEquals(trueParameters, openedModel);
    }

    @Test
    @Ignore
    public void testSaveModelText() {
        assertTrue(false);
    }

    @Test
    public void testLoadModelSerialized() {

        val model = compositionalLM.loadModelSerialized(TRUE_FILENAME);
        assertEquals(trueParameters, model);
    }

    @Test
    @Ignore
    public void testLoadModelText() {
        assertTrue(false);
    }
}
