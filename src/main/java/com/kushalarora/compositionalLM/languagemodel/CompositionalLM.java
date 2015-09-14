package com.kushalarora.compositionalLM.languagemodel;


import com.google.common.base.Function;
import com.google.common.collect.Lists;
import com.kushalarora.compositionalLM.caching.CacheFactory;
import com.kushalarora.compositionalLM.caching.CacheWrapper;
import com.kushalarora.compositionalLM.documentprocessor.DocumentProcessorFactory;
import com.kushalarora.compositionalLM.lang.*;
import com.kushalarora.compositionalLM.model.CompositionalGrammar;
import com.kushalarora.compositionalLM.model.Derivatives;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.optimizer.AbstractOptimizer;
import com.kushalarora.compositionalLM.optimizer.OptimizerFactory;
import com.kushalarora.compositionalLM.options.ArgParser;
import com.kushalarora.compositionalLM.options.Options;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.io.RuntimeIOException;
import lombok.Getter;
import lombok.NonNull;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.log4j.PropertyConfigurator;

import javax.annotation.Nullable;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by karora on 6/12/15.
 */
@Slf4j
@Getter
public class CompositionalLM {

    private final Options op;
    private final CompositionalGrammar compGrammar;
    private final DocumentProcessorFactory docProcessorFactory;
    private final Model model;
    CacheWrapper<Sentence, IInsideOutsideScore> cache;


    public CompositionalLM(Model model, Options op) throws IOException {
        this.model = model;
        this.compGrammar = new CompositionalGrammar(model, op);
        this.op = op;
        cache = new CacheFactory(model).getCache(op);

        docProcessorFactory =
                new DocumentProcessorFactory(
                        op,
                        new TokenizerFactory(
                                op, model.getGrammar()));
    }

    @SneakyThrows
    public void train() {
        // List of validation documents. Documents are list of sentences.
        List<List<Sentence>> validIterators = new ArrayList<List<Sentence>> ();
        for (String validFile : op.trainOp.validationFiles) {
            validIterators.add(
                    Lists.newArrayList(docProcessorFactory
                            .getDocumentProcessor(validFile)));
        }

        // List of training documents.
        List<List<Sentence>> trainIterators = new ArrayList<List<Sentence>>();
        for (String trainFile : op.trainOp.trainFiles) {
            trainIterators.add(Lists.newArrayList(docProcessorFactory
                    .getDocumentProcessor(trainFile)));
        }

        // Optimizer with scorer, derivative calculator and saver as argument.
        AbstractOptimizer<Sentence, Derivatives> optimizer =
                OptimizerFactory.getOptimizer(op, model,
                        new Function<Sentence, Double>() {
                            @Nullable
                            public Double apply(Sentence data) {                // scorer
                                IInsideOutsideScore preScore = cache.get(data);
                                CompositionalGrammar.CompositionalInsideOutsideScore score =
                                        compGrammar.computeScore(data,
                                                preScore);
                                return score.getSentenceScore();
                            }
                        },
                        new Function<Sentence, Derivatives>() {
                            @Nullable
                            public Derivatives apply(@Nullable Sentence sample) {              // derivative calculator
                                Derivatives derivatives = new Derivatives(op, model, sample);
                                IInsideOutsideScore preScore = cache.get(sample);
                                CompositionalGrammar.CompositionalInsideOutsideScore score =
                                        compGrammar.computeScore(sample,
                                                preScore);
                                derivatives.calcDerivative(score);
                                return derivatives;
                            }
                        },
                        new Function<Void, Void>() {
                            @Nullable
                            public Void apply(@Nullable Void input) {           // saver
                                saveModelSerialized(op.modelOp.outFilename);
                                return null;
                            }
                        });

        // Fit training data with validation on validation file.
        optimizer.fit(trainIterators, validIterators);

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

        Model model = loadModel(op);
        if (model == null) {
            if (!op.train) {
                throw new RuntimeException("You must specify model file using -model argument");
            }

            @NonNull IGrammar grammar = GrammarFactory.getGrammar(op);
            model = new Model(op.modelOp.dimensions, grammar);
        }
        final CompositionalLM cLM = new CompositionalLM(model, op);

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
        } else if (op.nbestRescore) {
            cLM.nbestList();
        } else if (op.parse) {
            cLM.parse();
        } // end processing if statement


    }   // end of main


}
