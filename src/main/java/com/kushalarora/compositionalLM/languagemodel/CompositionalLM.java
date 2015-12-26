package com.kushalarora.compositionalLM.languagemodel;


import com.google.common.base.Function;
import com.google.common.collect.Lists;
import com.kushalarora.compositionalLM.caching.CacheFactory;
import com.kushalarora.compositionalLM.caching.CacheWrapper;
import com.kushalarora.compositionalLM.derivatives.Derivatives;
import com.kushalarora.compositionalLM.documentprocessor.DocumentProcessorFactory;
import com.kushalarora.compositionalLM.documentprocessor.DocumentProcessorWrapper;
import com.kushalarora.compositionalLM.lang.*;
import com.kushalarora.compositionalLM.model.CompositionalGrammar;
import com.kushalarora.compositionalLM.model.CompositionalInsideOutsideScore;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.optimizer.AbstractOptimizer;
import com.kushalarora.compositionalLM.optimizer.OptimizerFactory;
import com.kushalarora.compositionalLM.options.ArgParser;
import com.kushalarora.compositionalLM.options.Options;
import com.kushalarora.compositionalLM.utils.Visualization;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.io.RuntimeIOException;
import edu.stanford.nlp.util.IntTuple;
import lombok.Getter;
import lombok.NonNull;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.log4j.PropertyConfigurator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.annotation.Nullable;

import java.io.File;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.regex.Pattern;

/**
 * Created by karora on 6/12/15.
 */
@Slf4j
@Getter
public class CompositionalLM {

    private final Options op;
    private final CompositionalGrammar compGrammar;
    private final IGrammar grammar;
    private final DocumentProcessorFactory docProcessorFactory;
    private final Model model;
    CacheWrapper<Sentence, IInsideOutsideScore> cache;


    public CompositionalLM(Model model, IGrammar grammar, Options op) throws IOException {
        this.model = model;
        this.grammar = grammar;
        this.compGrammar = new CompositionalGrammar(model, op);
        this.op = op;
        cache = new CacheFactory(grammar).getCache(op);

        docProcessorFactory =
                new DocumentProcessorFactory(
                        op,
                        new TokenizerFactory(
                                op, grammar));
    }

    @SneakyThrows
    public void train() {
        // List of validation documents. Documents are list of sentences.

        final Map<Integer, String> trainIndexSet = new HashMap<Integer, String>();

        // Optimizer with scorer, derivative calculator and saver as argument.
        AbstractOptimizer<Sentence, Derivatives> optimizer =
                OptimizerFactory.getOptimizer(op, model,
                        new Function<Sentence, Double>() {
                            @Nullable
                            public Double apply(Sentence data) {                // scorer
                                IInsideOutsideScore preScore = cache.get(data);
                                CompositionalInsideOutsideScore score =
                                        compGrammar.getScore(data,
                                                preScore);
                                return score.getSentenceScore();
                            }
                        },
                        new Function<Sentence, Derivatives>() {
                            @Nullable
                            public Derivatives apply(@Nullable Sentence sentence) {              // derivative calculator
                                Derivatives derivatives = new Derivatives(op,
                                        model.getDimensions(), model.getVocabSize(), sentence);
                                IInsideOutsideScore preScore = cache.get(sentence);
                                CompositionalInsideOutsideScore score =
                                        compGrammar.getScore(sentence,
                                                preScore);
                                derivatives.calcDerivative(model, score);
                                return derivatives;
                            }
                        },
                        new Function<IntTuple, Void>() {
                            @Nullable
                            public Void apply(@Nullable IntTuple tuple) {           // saver
                                String[] str = op.modelOp.outFilename.split(Pattern.quote("."));
                                str[0] = String.format("%s-%d-%d", str[0], tuple.get(0), tuple.get(1));
                                String outFilename = String.join(".", str);
                                saveModelSerialized(outFilename);
                                return null;
                            }
                        });

        DocumentProcessorWrapper<Sentence> docProcessor =
                docProcessorFactory.getDocumentProcessor();

        List<Iterator<Sentence>> trainSentIter = Lists.newArrayList();
        for (String filename : op.trainOp.trainFiles) {
            trainSentIter.add(docProcessor.getIterator(filename));
        }

        List<Iterator<Sentence>> validSentIter = Lists.newArrayList();
        for (String filename : op.trainOp.validationFiles) {
            validSentIter.add(docProcessor.getIterator(filename));
        }

        // Fit training data with validation on validation file.
        optimizer.fit(trainSentIter, validSentIter);

        if (op.trainOp.saveVisualization) {
            visualize(op.trainOp.visualizationFilename);
        }

        saveModelSerialized(op.modelOp.outFilename);
        // Closing cache. Ecache doesn't do eternal caching
        // until and unless closed
        cache.close();

    }

    public void parse() {
       /* for (String filename : op.testOp.parseFiles) {
            DocumentIterator docProcessor = docProcessorFactory
                    .getDocumentIterator(filename);
            for (Sentence sentence : docProcessor) {
                compGrammar.parse(sentence);
            }

        }*/

    }

    public void nbestList() {
       /* for (String filename : op.testOp.nbestFiles) {
            DocumentIterator docProcessor = docProcessorFactory
                    .getDocumentIterator(filename);
            // TODO:: Figure this out

        }*/

    }

    public void contrastiveEntropy() throws IOException {
        double logScore = 0f;
        PrintWriter writer = new PrintWriter(new File(op.testOp.outputFile), "UTF-8");
        for (String testFile : op.testOp.testFiles) {
            DocumentProcessorWrapper<Sentence> documentProcessor =
                    docProcessorFactory.getDocumentProcessor();
            Iterator<Sentence> testIter = documentProcessor.getIterator(testFile);

            while (testIter.hasNext()) {
                Sentence data = testIter.next();
                IInsideOutsideScore preScore = cache.get(data);
                CompositionalInsideOutsideScore score =
                        compGrammar.getScore(data,
                                preScore);
                Double logP = score.getSentenceScore();
                writer.println(score.getSentence());
                writer.println(String.format("Length: %d, logProp: %.4f",
                                             score.getSentence().size(), score.getSentenceScore()));
                log.info(String.format("Length: %d, logProp: %.4f",
                                       score.getSentence().size(), score.getSentenceScore()));
                logScore += logP;
            }
        }
        writer.println(String.format("Total logProb:%.4f", logScore));
        log.info("Total logProb:{}", logScore);
        writer.close();

    }

    @SneakyThrows
    public void visualize(String filename) {
        // List of training documents.
        List<List<Sentence>> trainIterators = new ArrayList<List<Sentence>>();
        for (String trainFile : op.trainOp.trainFiles) {
            trainIterators.add(Lists.newArrayList(docProcessorFactory
                    .getDocumentProcessor().getIterator(filename)));
        }

        final Map<Integer, String> trainIndexSet = new HashMap<Integer, String>();
        for (List<Sentence> trainList : trainIterators) {
            for (Sentence sent : trainList) {
                for (Word word : sent) {
                    if (!trainIndexSet.containsKey(word.getIndex())) {
                        trainIndexSet.put(word.getIndex(), word.getSignature());
                    }
                }
            }
        }

        INDArray embeddedWords = Nd4j.zeros(model.getDimensions(),
                trainIndexSet.size());
        List<String> words = new ArrayList<String>();
        int i = 0;
        INDArray X = model.getParams().getX();
        for (Map.Entry<Integer, String> entrySet : trainIndexSet.entrySet()) {
            embeddedWords.putColumn(i++, X.getColumn(entrySet.getKey()));
            words.add(entrySet.getValue());
        }

        try {
            Visualization.saveTNSEVisualization(
                    filename,
                    embeddedWords,
                    words);
        } catch (IOException e) {
            log.error("Failed visualization", e);
        }

    }

    /**
     * Saves model as a serialized file to the given filename
     *
     * @param filename Name of file to be saved
     */
    public void saveModelSerialized(String filename) {
        try {
            log.info("Writing model in serialized format to file: {}", filename);
            ObjectOutputStream out = IOUtils.writeStreamFromString(filename);
            out.writeObject(model);
            out.close();
        } catch (IOException e) {
            log.error("Filename {} couldn't be opened for writing", filename);
            throw new RuntimeIOException(e);
        }
    }

    /**
     * Save model as a text file to the given filename
     *
     * @param filename Name of the file to be saved
     */
    public void saveModelText(String filename) {
        // TODO:: Figure this out how to write the file to text
    }

    /**
     * Load serialized model for parsing or re-ranking
     *
     * @param filename Model serialized file
     */
    public static Model loadModelSerialized(String filename) {
        log.info("Loading model from serialized file: {}", filename);
        try {
            ObjectInputStream in = IOUtils.readStreamFromString(filename);
            Object o = in.readObject();
            if (o instanceof Model) {
                return (Model) o;
            }
            throw new ClassCastException(
                    String.format("Wanted class Model, got %s", o.getClass().toString()));
        } catch (IOException e) {
            throw new RuntimeIOException(
                    String.format("Model file not found: %s", filename), e);
        } catch (ClassNotFoundException e) {
            throw new RuntimeIOException(
                    String.format("Invalid model file: %s", filename), e);
        }
    }

    /**
     * Load model text file for parsing or re-ranking
     *
     * @param filename Model text file
     */
    public static Model loadModelText(String filename) {
        // TODO:: Load model once figured out.
        return null;
    }

    public static Model loadModel(Options op) {
        val type = op.modelOp.inType;
        String filename = op.modelOp.inFilename;

        // Return null if no filename or file type specified
        if (type == null || filename == null) {
            return null;
        }

        Model model = null;
        if (type.equals(Options.FileType.TEXT)) {
            model = loadModelText(filename);
        } else if (type.equals(Options.FileType.SERIALIZED)) {
            model = loadModelSerialized(filename);
        }
        return model;
    }

    /**
     * TODO:: Explain the usage as done by  parser
     *
     * @param args
     */
    public static void main(String[] args) throws Exception {

        PropertyConfigurator.configure("log4j.properties");
        Options op = ArgParser.parseArgs(args);
        log.info("Options: {}", op);

        @NonNull IGrammar grammar = GrammarFactory.getGrammar(op);

        Model model;
        if (!op.train) {
            model =loadModel(op);
            if (model == null) {
                throw new RuntimeException("You must specify model file using -model argument");
            }
        } else {
            model = new Model(op.modelOp.dimensions,
                              grammar.getVocabSize(),
                              op.grammarOp.grammarType);
        }

        final CompositionalLM cLM = new CompositionalLM(model, grammar, op);

        Runtime.getRuntime().addShutdownHook(new Thread() {
            @Override
            public void run() {
                log.error("Exiting Closing Cache");
                // Ecache needs to be closed in all cases
                cLM.cache.close();
            }
        });

        if (op.train) {
            log.info("starting training");
            cLM.train();
        } else if (op.visualize) {
            cLM.visualize(op.testOp.visualizationFile);
        } else if (op.nbestRescore) {
            cLM.nbestList();
        } else if (op.parse) {
            cLM.parse();
        } else if (op.test) {
            cLM.contrastiveEntropy();
        } // end processing if statement


    }   // end of main


}
