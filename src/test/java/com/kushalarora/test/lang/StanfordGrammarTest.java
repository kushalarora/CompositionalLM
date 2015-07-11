package com.kushalarora.test.lang;

import static com.kushalarora.compositionalLM.lang.GrammarFactory.*;

import com.kushalarora.compositionalLM.lang.GrammarFactory;
import com.kushalarora.compositionalLM.lang.IInsideOutsideScores;
import com.kushalarora.compositionalLM.lang.StanfordGrammar;
import com.kushalarora.compositionalLM.lang.Word;
import com.kushalarora.compositionalLM.options.Options;
import lombok.val;
import org.apache.commons.io.FileUtils;
import org.junit.BeforeClass;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by karora on 6/25/15.
 */
public class StanfordGrammarTest {

    public static String GRAMMAR_RELATIVE_FILE_PATH = "src/test/resources/englishPCFG.ser.gz";
    public static StanfordGrammar sg;
    public static List<Word> defaultSentence;

    @BeforeClass
    public static void setUpClass() {
        String absoluteFilePath = FileUtils
                .getFile(GRAMMAR_RELATIVE_FILE_PATH)
                .getAbsolutePath();
        Options op = new Options();
        sg = (StanfordGrammar)getGrammar(GrammarFactory.GrammarType.STANFORD_GRAMMAR,
                                         absoluteFilePath, op);

        String[] sent = {"This", "is", "just", "a", "test", "."};
        for (String str : sent) {
            int index = (int)Math.random() * (sg.getVocabSize() + 1);
            defaultSentence.add(new Word(str, index));
        }
    }

}
