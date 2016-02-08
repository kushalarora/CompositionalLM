package com.kushalarora.compositionalLM.languagemodel;


import com.google.common.base.Function;
import com.google.common.collect.Lists;
import com.kushalarora.compositionalLM.derivatives.Derivatives;
import com.kushalarora.compositionalLM.documentprocessor.DocumentProcessorFactory;
import com.kushalarora.compositionalLM.documentprocessor.DocumentProcessorWrapper;
import com.kushalarora.compositionalLM.lang.*;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.optimizer.AbstractOptimizer;
import com.kushalarora.compositionalLM.optimizer.OptimizerFactory;
import com.kushalarora.compositionalLM.options.ArgParser;
import com.kushalarora.compositionalLM.options.Options;
import com.kushalarora.compositionalLM.utils.Parallelizer;
import com.kushalarora.compositionalLM.utils.Visualization;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.io.RuntimeIOException;
import edu.stanford.nlp.util.IntTuple;
import lombok.Getter;
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
    private final StanfordCompositionalGrammar grammar;
    private final DocumentProcessorFactory docProcessorFactory;
    private final Model model;
    private Parallelizer parallelizer;

    public CompositionalLM(StanfordCompositionalGrammar grammar, Options op, Model model,
                           Parallelizer parallelizer)
            throws IOException {
        this.grammar = grammar;
        this.model = model;
        this.op = op;
        docProcessorFactory = new DocumentProcessorFactory(
                                op, new TokenizerFactory( op, grammar));
        this.parallelizer = parallelizer;
    }

    @SneakyThrows
    public void train() {
        DocumentProcessorWrapper<Sentence> docProcessor =
                docProcessorFactory.getDocumentProcessor();

        List<List<Sentence>> trainSentList = Lists.newArrayList();
        List<List<Sentence>> trainDistSentList = Lists.newArrayList();
        for (String filename : op.trainOp.trainFiles) {
            trainSentList.add(Lists.newArrayList(docProcessor.getIterator(filename)));
            trainDistSentList.add(Lists.newArrayList(
                    docProcessor.getIterator(getDistortedFilename(filename))));
        }

        List<List<Sentence>> validSentList = Lists.newArrayList();
        List<List<Sentence>> validDistSentList = Lists.newArrayList();
        for (String filename : op.trainOp.validationFiles) {
            validSentList.add(Lists.newArrayList(docProcessor.getIterator(filename)));
            validDistSentList.add(Lists.newArrayList(
                    docProcessor.getIterator(getDistortedFilename(filename))));

        }


        List<List<StanfordCompositionalInsideOutsideScore>> validScoreFileList = Lists.newArrayList();
        for (final List<Sentence> validList : validSentList) {
            final int validListSize = validList.size();
            final List<StanfordCompositionalInsideOutsideScore> validScoreList =
                    new ArrayList<StanfordCompositionalInsideOutsideScore>();

            Function<Integer, Void> validScorerFunc =
                    new Function<Integer, Void>() {
                        @Nullable
                        public Void apply(@Nullable Integer index) {
                            synchronized (validScoreList) {
                                validScoreList.add(
                                        new StanfordCompositionalInsideOutsideScore(
                                                validList.get(index),
                                                model.getDimensions(),
                                                grammar.getVocabSize(),
                                                false
                                        ));
                            }
                            return null;
                        }
                    };
            if (op.trainOp.parallel) {
                parallelizer.parallelizer(0, validListSize, validScorerFunc);
            } else {
                for (int index = 0; index < validListSize; index++) {
                    validScorerFunc.apply(index);
                }
            }
            validScoreFileList.add(validScoreList);
        }

            // Optimizer with scorer, derivative calculator and saver as argument.
         AbstractOptimizer<StanfordCompositionalInsideOutsideScore, Derivatives> optimizer =
                OptimizerFactory.getOptimizer(op, model,
                         new Function<StanfordCompositionalInsideOutsideScore, Double>() {
                                @Nullable
                                // scorer
                                public Double apply(StanfordCompositionalInsideOutsideScore score) {
                                    return ((StanfordCompositionalInsideOutsideScore)
                                            grammar.getInsideScore(score.getSentence(), true))
                                            .getSentenceScore();
                                }
                         },
                         new Function<StanfordCompositionalInsideOutsideScore, Derivatives>() {
                                @Nullable
                                // derivative calculator
                                public Derivatives apply(@Nullable StanfordCompositionalInsideOutsideScore score) {
                                Derivatives derivatives = new Derivatives(op,
                                        model, score);
                                derivatives.calcDerivative();
                                return derivatives;
                             }
                         },
                         new Function<IntTuple, Void>() {
                             @Nullable
                             // saver
                             public Void apply(@Nullable IntTuple tuple) {
                                 String[] str = op.modelOp.outFilename.split(Pattern.quote("."));
                                 str[0] = String.format("%s-%d-%d", str[0], tuple.get(0), tuple.get(1));
                                 String outFilename = String.join(".", str);
                                 saveModelSerialized(outFilename);
                                 return null;
                             }
                         }, parallelizer);

        int EMIter = 0;
        // TODO: Add early stopping logic.
        while (EMIter < op.trainOp.maxEMEpochs) {

            // E Step
            List<List<StanfordCompositionalInsideOutsideScore>> trainScoreFileList = Lists.newArrayList();
            for (final List<Sentence> trainList : trainSentList) {
                final int trainListSize = trainList.size();
                final List<StanfordCompositionalInsideOutsideScore> trainScoreList =
                        new ArrayList<StanfordCompositionalInsideOutsideScore>();

                Function<Integer, Void> trainScorerFunc =
                        new Function<Integer, Void>() {
                            @Nullable
                            public Void apply(@Nullable Integer index) {
                                StanfordCompositionalInsideOutsideScore score =
                                (StanfordCompositionalInsideOutsideScore)
                                        grammar.getScore(trainList.get(index));
                                synchronized (trainScoreList) {
                                    trainScoreList.add(score);
                                }
                                return null;
                            }
                        };
                if (op.trainOp.parallel) {
                    parallelizer.parallelizer(0, trainListSize, trainScorerFunc);
                } else {
                    for (int index = 0; index < trainListSize; index++) {
                        trainScorerFunc.apply(index);
                    }
                }
                trainScoreFileList.add(trainScoreList);
            }

            // M Step
            // Fit training data with validation on validation file.
            optimizer.fit(trainScoreFileList, validScoreFileList);

            log.info("EMIter#: {}, bestValidationScore => {}",
                        EMIter, optimizer.getBestValidationScore());

            double combinedContEntropyScore = 0;
            int combinedTrainSize = 0;
            for (int i = 0; i < trainSentList.size(); i++) {
                final List<Sentence> trainList = trainSentList.get(i);
                final List<Sentence> trainDistList = trainDistSentList.get(i);
                int contEntropyScore = 0;
                Function<Integer, Double> testFunc = new Function<Integer, Double>() {
                    @Nullable
                    public Double apply(@Nullable Integer i) {
                        return crossEntropySent(trainList.get(i),
                                                trainDistList.get(i));
                    }
                };
                final int trainListSize = trainList.size();

                if (op.trainOp.parallel) {
                    List<Future<List<Double>>> contEntropyFunc =
                            parallelizer.parallelizer(0, trainListSize, testFunc);

                    for (Future<List<Double>> future : contEntropyFunc) {
                        List<Double> scoreList = future.get();
                        for (double score : scoreList) {
                            contEntropyScore += score;
                        }
                    }
                } else {
                    for (int j = 0; j < trainListSize; j++) {
                        double score = testFunc.apply(j);
                        contEntropyScore += score;
                    }
                }
                combinedTrainSize += trainListSize;
                combinedContEntropyScore += contEntropyScore;
                log.info("$ContEnt$ Avg Contrastive Train Entropy for batch#{} : {}",
                            i, contEntropyScore/trainListSize);
            }

            log.info("$ContEnt$ Avg Contrastive Train Entropy : {}",
                        combinedContEntropyScore/combinedTrainSize);


            EMIter++;
        }

        if (op.trainOp.saveVisualization) {
            visualize(op.trainOp.visualizationFilename);
        }

        saveModelSerialized(op.modelOp.outFilename);
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

    public void entropy() throws IOException, ExecutionException, InterruptedException {
        double logScore = 0f;
        final PrintWriter writer = new PrintWriter(new File(op.testOp.outputFile), "UTF-8");
        int testFileIdx = 0;
        DocumentProcessorWrapper<Sentence> documentProcessor =
                docProcessorFactory.getDocumentProcessor();
        for (String testFile : op.testOp.testFiles) {
            float logScoreFile = 0f;
            long epochTestfileTime = System.currentTimeMillis();
            Iterator<Sentence> testIter = documentProcessor.getIterator(testFile);

            while (testIter.hasNext()) {
                final List<Sentence> testList = new ArrayList<Sentence>();

                for (int idx = 0; idx < op.testOp.testBatchSize && testIter.hasNext(); idx++) {
                    testList.add(testIter.next());
                }

                int testBatchSize = testList.size();

                Function<Integer, Double> testFunc = new Function<Integer, Double>() {
                    @Nullable
                    public Double apply(@Nullable Integer integer) {
                        StanfordCompositionalInsideOutsideScore score =
                                (StanfordCompositionalInsideOutsideScore)
                                        grammar.getInsideScore(testList.get(integer), true);
                        Sentence sentence = score.getSentence();
                        Double logP = score.getSentenceScore();
                        log.info(String.format("Length: %d, logProp: %.4f", sentence.size(), logP));
                        synchronized (writer) {
                            writer.println(sentence);
                            writer.println(String.format("Length: %d, logProp: %.4f", sentence.size(), logP));
                        }
                        return logP;
                    }
                };

                if (op.trainOp.parallel) {
                    List<Future<List<Double>>> testScoreFutures =
                            parallelizer.parallelizer(0, testBatchSize, testFunc);

                    for (Future<List<Double>> future : testScoreFutures) {
                        List<Double> scoreList = future.get();
                        for (double score : scoreList) {
                            logScoreFile += score;
                        }
                    }
                } else {
                    for (int i = 0; i < testBatchSize; i++) {
                        double score = testFunc.apply(i);
                        logScoreFile += score;
                    }
                }
            }
            double estimatedTestfileTime = System.currentTimeMillis() - epochTestfileTime;
            log.info("$Testing$:: Constrastive Entropy calculated for file Idx:{}, time: {} => {}",
                        testFileIdx, estimatedTestfileTime, logScoreFile);
            logScore += logScoreFile;
            testFileIdx++;
        }

        writer.println(String.format("Total logProb:%.4f", logScore));
        log.info("Total logProb:{}", logScore);
        writer.close();
    }

    public void crossEntropy() throws IOException, ExecutionException, InterruptedException {
        final PrintWriter writer = new PrintWriter(new File(op.testOp.outputFile), "UTF-8");
        int testFileIdx = 0;
        DocumentProcessorWrapper<Sentence> documentProcessor =
                docProcessorFactory.getDocumentProcessor();
        for (String testFile : op.testOp.testFiles) {
            String testDistractedFile = getDistortedFilename(testFile);
            float logScoreFile = 0f;
            long epochTestfileTime = System.currentTimeMillis();
            Iterator<Sentence> testIter = documentProcessor.getIterator(testFile);
            Iterator<Sentence> testDistIter = documentProcessor.getIterator(testDistractedFile);


            while (testIter.hasNext()) {
                final List<Sentence> testList = new ArrayList<Sentence>();
                final List<Sentence> testDistList = new ArrayList<Sentence>();

                for (int idx = 0; idx < op.testOp.testBatchSize && testIter.hasNext(); idx++) {
                    testList.add(testIter.next());
                    testDistList.add(testDistIter.next());
                }

                double contEntropyScore = 0;
                int testBatchSize = testList.size();
                Function<Integer, Double> testFunc = new Function<Integer, Double>() {
                    @Nullable
                    public Double apply(@Nullable Integer i) {
                        return crossEntropySent(writer,
                                                testList.get(i),
                                                testDistList.get(i));
                    }
                };

                if (op.trainOp.parallel) {
                    List<Future<List<Double>>> testScoreFutures =
                            parallelizer.parallelizer(0, testBatchSize, testFunc);

                    for (Future<List<Double>> future : testScoreFutures) {
                        List<Double> scoreList = future.get();
                        for (double score : scoreList) {
                            contEntropyScore += score;
                        }
                    }
                } else {
                    for (int j = 0; j < testBatchSize; j++) {
                        double score = testFunc.apply(j);
                        contEntropyScore += score;
                    }
                }

            }
            double estimatedTestfileTime = System.currentTimeMillis() - epochTestfileTime;
            log.info("$Testing$:: Constrastive Entropy calculated for file Idx:{}, time: {} => {}",
                     testFileIdx, estimatedTestfileTime, logScoreFile);
            testFileIdx++;
        }
    }

    private String getDistortedFilename(String filename) {
        double errPct = op.inputOp.errPct;
        double originalPct = 100 - errPct;
        double eachErrPct = errPct / 2;
        return String.format("%s.deformed-%.1f-%.1f-%.1f",
                             filename, originalPct,
                             eachErrPct, eachErrPct);
    }

    private double crossEntropySent(final PrintWriter writer, final Sentence sentence, final Sentence distSentence) {
        double scoreSent =
                ((StanfordCompositionalInsideOutsideScore)
                        grammar.getInsideScore(sentence, true))
                            .getSentenceScore();
        double scoreDistortedSent =
                ((StanfordCompositionalInsideOutsideScore)
                        grammar.getInsideScore(distSentence, true))
                            .getSentenceScore();
        double contEntropy =  scoreSent - scoreDistortedSent;
        log.info(String.format("Sentence#%d[%d] contrastiveEntropy => %.4f",
                                    sentence.getIndex(),  sentence.size(), contEntropy));

        if (writer != null) {
            synchronized (writer) {
                writer.println(sentence);
                writer.println(String.format("Length: %d, logProp: %.4f", sentence.size(), contEntropy));
            }
        }
        return contEntropy;
    }

    private double crossEntropySent(final Sentence sentence, final Sentence distSentence) {
        return crossEntropySent(null, sentence, distSentence);
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

        Model model = null;
        if (!op.train) {
            model = loadModel(op);
            if (model == null) {
                throw new RuntimeException("You must specify model file using -model argument");
            }
        }

        Parallelizer parallelizer = new Parallelizer(op, 1);

        StanfordCompositionalGrammar grammar;
        if (model != null) {
            grammar = (StanfordCompositionalGrammar)GrammarFactory.getGrammar(op, model, parallelizer);
        } else {
            grammar = (StanfordCompositionalGrammar)GrammarFactory.getGrammar(op, parallelizer);
            model = grammar.getModel();
        }

        final CompositionalLM cLM = new CompositionalLM(grammar, op, model, parallelizer);

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
            cLM.entropy();
        } // end processing if statement


    }   // end of main


}
