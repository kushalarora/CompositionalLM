package com.kushalarora.compositionalLM.languagemodel;


import com.google.common.collect.Lists;
import com.kushalarora.compositionalLM.lang.*;
import com.kushalarora.compositionalLM.model.*;
import com.kushalarora.compositionalLM.optimizer.AbstractSGDOptimizer;
import com.kushalarora.compositionalLM.options.ArgParser;
import com.kushalarora.compositionalLM.options.Options;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.io.RuntimeIOException;
import lombok.NonNull;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.log4j.PropertyConfigurator;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by karora on 6/12/15.
 */
@Slf4j
public class CompositionalLM {

    private final Options op;
    private final CompositionalGrammar compGrammar;
    private final DocumentProcessorFactory docProcessorFactory;
    private final Model model;

    public CompositionalLM(Model model, Options op) {
        this.model = model;
        this.compGrammar = new CompositionalGrammar(model, op);
        this.op = op;

        docProcessorFactory =
                new DocumentProcessorFactory(
                        op,
                        new TokenizerFactory(
                                op, model.getGrammar()));
    }

    @SneakyThrows
    public void train() {
        List<List<Word>> validSentences = new ArrayList<List<Word>>();
        for (String validFile : op.trainOp.validationFiles) {
            DocumentProcessorWrapper docProcessor =
                    docProcessorFactory
                            .getDocumentProcessor(validFile);
            validSentences.addAll(
                    Lists.<List<Word>>newArrayList(
                            docProcessor));
        }

        for (String trainFile : op.trainOp.trainFiles) {
            int sentenceCount = 0;
            DocumentProcessorWrapper trainDocProcessor =
                    docProcessorFactory
                            .getDocumentProcessor(trainFile);


            final Derivatives dv = new Derivatives(model);
            AbstractSGDOptimizer<List<Word>> optimizer =
                    new AbstractSGDOptimizer<List<Word>>(op) {
                        @Override
                        public double getValidationScore(List<Word> data) {
                            CompositionalGrammar.CompositionalInsideOutsideScore score =
                                    compGrammar.computeScore(data,
                                            model.getGrammar().computeScore(data));
                            return score.getSentenceScore();
                        }

                        @Override
                        public void saveModel() {
                            saveModelSerialized(op.modelOp.outFilename);
                        }

                        public IParameterDerivatives calcDerivative(List<Word> sample) {
                            Derivatives derivatives = new Derivatives(model);

                            CompositionalGrammar.CompositionalInsideOutsideScore score =
                                    compGrammar.computeScore(sample,
                                            model.getGrammar().computeScore(sample));

                            return derivatives.calcDerivative(sample, score);
                        }

                        public IParameter getParams() {
                            return model.getParams();
                        }

                        public void derivativeAccumulator(IParameterDerivatives derivatives) {

                            dv.add(derivatives);
                        }

                        public IParameterDerivatives getAccumulatedDerivative() {

                            return dv;
                        }

                        public void flushDerivaiveAccumulator() {

                            dv.clear();
                        }
                    };

            optimizer.fit(
                    Lists.newArrayList(trainDocProcessor),
                    validSentences);
        }

    }

    public void parse() {
        for (String filename : op.testOp.parseFiles) {
            DocumentProcessorWrapper docProcessor = docProcessorFactory
                    .getDocumentProcessor(filename);
            for (List<Word> sentence : docProcessor) {
                compGrammar.parse(sentence);
            }

        }

    }

    public void nbestList() {
        for (String filename : op.testOp.nbestFiles) {
            DocumentProcessorWrapper docProcessor = docProcessorFactory
                    .getDocumentProcessor(filename);
            // TODO:: Figure this out

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
        if (type.equals(Options.FILE_TYPE.TEXT)) {
            model = loadModelText(filename);
        } else if (type.equals(Options.FILE_TYPE.SERIALIZED)) {
            model = loadModelSerialized(filename);
        }
        return model;
    }

    /**
     * TODO:: Explain the usage as done by  parser
     *
     * @param args
     */
    public static void main(String[] args) {
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
        CompositionalLM cLM = new CompositionalLM(model, op);

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
