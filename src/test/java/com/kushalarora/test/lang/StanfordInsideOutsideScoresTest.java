package com.kushalarora.test.lang;

import com.kushalarora.compositionalLM.lang.GrammarFactory;
import com.kushalarora.compositionalLM.lang.Sentence;
import com.kushalarora.compositionalLM.lang.StanfordGrammar;
import com.kushalarora.compositionalLM.lang.Word;
import com.kushalarora.compositionalLM.options.Options;
import com.kushalarora.compositionalLM.utils.Parallelizer;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.configuration.ConfigurationException;
import org.apache.commons.io.FileUtils;
import org.apache.log4j.PropertyConfigurator;
import org.junit.BeforeClass;

import static com.kushalarora.compositionalLM.lang.GrammarFactory.getGrammar;

/**
 * Created by karora on 6/25/15.
 */


@Slf4j
public class StanfordInsideOutsideScoresTest {

    private int length;

    private static int numStates;
    private static Sentence defaultSentence;
    private static StanfordGrammar sg;


    @BeforeClass
    public static void setUpClass() throws ConfigurationException {
        val filePath = FileUtils.getFile("src/resources/wsjPCFG.ser.gz")
                .getAbsolutePath();

        Options op = new Options();
        op.grammarOp.grammarType = GrammarFactory.GrammarType.STANFORD_GRAMMAR;
        op.grammarOp.filename = filePath;
        sg = (StanfordGrammar) getGrammar(op, new Parallelizer(op, 1));
        numStates = sg.getNumStates();
        PropertyConfigurator.configure("log4j.properties");

        defaultSentence = new Sentence(0);
        String[] sent = {"This", "is", "just", "a", "test", "."};
        for (String str : sent) {
            int index = (int) Math.random() * (sg.getVocabSize() + 1);
            defaultSentence.add(new Word(str, index));
        }
    }


}
