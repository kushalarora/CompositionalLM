package com.kushalarora.test.lang;

import com.kushalarora.compositionalLM.lang.GrammarFactory;
import com.kushalarora.compositionalLM.lang.Word;
import com.kushalarora.compositionalLM.lang.stanford.StanfordGrammar;
import com.kushalarora.compositionalLM.options.Options;
import org.apache.commons.configuration.ConfigurationException;
import org.apache.commons.io.FileUtils;
import org.junit.BeforeClass;

import java.util.List;

import static com.kushalarora.compositionalLM.lang.GrammarFactory.getGrammar;

/**
 * Created by karora on 6/25/15.
 */
public class StanfordGrammarTest {

    public static String GRAMMAR_RELATIVE_FILE_PATH = "src/resources/englishPCFG.ser.gz";
    public static StanfordGrammar sg;
    public static List<Word> defaultSentence;

    @BeforeClass
    public static void setUpClass() throws ConfigurationException {
        String absoluteFilePath = FileUtils
                .getFile(GRAMMAR_RELATIVE_FILE_PATH)
                .getAbsolutePath();
        Options op = new Options();
        op.grammarOp.grammarType = GrammarFactory.GrammarType.STANFORD_GRAMMAR;
        op.grammarOp.filename =  absoluteFilePath;
        sg = (StanfordGrammar)getGrammar(op);

        String[] sent = {"This", "is", "just", "a", "test", "."};
        for (String str : sent) {
            int index = (int)Math.random() * (sg.getVocabSize() + 1);
            defaultSentence.add(new Word(str, index));
        }
    }

}
