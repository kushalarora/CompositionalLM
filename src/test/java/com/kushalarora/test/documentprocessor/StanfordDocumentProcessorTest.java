package com.kushalarora.test.documentprocessor;

import com.kushalarora.compositionalLM.documentprocessor.StanfordDocumentProcessor;
import com.kushalarora.compositionalLM.lang.GrammarFactory;
import com.kushalarora.compositionalLM.lang.Sentence;
import com.kushalarora.compositionalLM.lang.TokenizerFactory;
import com.kushalarora.compositionalLM.lang.stanford.StanfordGrammar;
import com.kushalarora.compositionalLM.options.Options;
import com.sun.javafx.fxml.expression.Expression;
import org.apache.commons.configuration.ConfigurationException;
import org.apache.commons.io.FileUtils;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Iterator;

import static com.kushalarora.compositionalLM.lang.GrammarFactory.getGrammar;
import static junit.framework.TestCase.assertEquals;

/**
 * Created by arorak on 9/14/15.
 */
public class StanfordDocumentProcessorTest {
    public static String GRAMMAR_RELATIVE_FILE_PATH = "src/resources/englishPCFG.ser.gz";
    public static String FILENAME = "src/resources/documentProcessorMaxLengthTest.txt";
    public static StanfordGrammar sg;
    public static StanfordDocumentProcessor sDP;


    @BeforeClass
    public static void setUpClass() throws ConfigurationException {
        String absoluteFilePath = FileUtils
                .getFile(GRAMMAR_RELATIVE_FILE_PATH)
                .getAbsolutePath();
        Options op = new Options();
        op.grammarOp.grammarType = GrammarFactory.GrammarType.STANFORD_GRAMMAR;
        op.grammarOp.filename =  absoluteFilePath;
        op.grammarOp.maxLength = 10;
        sg = (StanfordGrammar)getGrammar(op);
        sDP = new StanfordDocumentProcessor(op, FILENAME, new TokenizerFactory(op, sg));
    }


    @Test
    public void testMaxLength() {
        Iterator<Sentence> it = sDP.iterator();

        ArrayList<Sentence> list = new ArrayList<Sentence>();

        while (it.hasNext()) {
            list.add(it.next());
        }

        assertEquals(2, list.size());

    }
}
