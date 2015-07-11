package com.kushalarora.compositionalLM.languagemodel;


import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.options.ParserOptions;
import com.kushalarora.compositionalLM.utils.ArgUtils;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.io.RuntimeIOException;
import lombok.extern.slf4j.Slf4j;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

/**
 * Created by karora on 6/12/15.
 */
@Slf4j
public class CompositionalLM {

    private Model model;

    public CompositionalLM(Model model) {
        this.model = model;
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
    public Model loadModelSerialized(String filename) {
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
    public Model loadModelText(String filename) {
        // TODO:: Load model once figured out.
        return null;
    }


    /**
     * TODO:: Explain the usage as done by  parser
     *
     * @param args
     */
    public static void main(String[] args) {
        int argIndex = 0;
        ParserOptions op = new ParserOptions();
        if (args.length < 1) {
            log.error("TODO: Write Error message about too few arguments");
        }

        while (argIndex < args.length) {
            if (args[argIndex].equalsIgnoreCase("-train")) {
                // Must be followed by a training file
                op.train = true;
                op.trainFiles = ArgUtils.getFileNameFromArg(args, argIndex);
                argIndex = ArgUtils.numSubArgs(args, argIndex);

            } else if (args[argIndex].equalsIgnoreCase("-nbest")) {
                // Must be followed by a text file with n best list generated from
                // lattice decoder to be reranked
                op.nbestRescore = true;
                op.nbestFiles = ArgUtils.getFileNameFromArg(args, argIndex);
                argIndex += ArgUtils.numSubArgs(args, argIndex);

            } else if (args[argIndex].equalsIgnoreCase("-validationFile")) {
                // Ignored if only testing.
                // If training must be followed by validation File
                op.validationFiles = ArgUtils.getFileNameFromArg(args, argIndex);
                argIndex += ArgUtils.numSubArgs(args, argIndex);

            } else if (args[argIndex].equalsIgnoreCase("-parse")) {
                // Followed by file to be parsed.
                op.parse = true;
                op.parseFiles = ArgUtils.getFileNameFromArg(args, argIndex);
                argIndex += ArgUtils.numSubArgs(args, argIndex);

            } else if (args[argIndex].equalsIgnoreCase("-grammarTextFile")) {
                // load grammar from text file.
                // following argument should be a path to text file
                op.grammarTypeFile = ParserOptions.FILE_TYPE.TEXT;
                String[] files = ArgUtils.getFileNameFromArg(args, argIndex);
                if (files.length > 1) {
                    throw new RuntimeException("Can't have multiple grammar files.");
                }
                op.grammarFilePath = files[0];
                argIndex++;

            } else if (args[argIndex].equalsIgnoreCase("-grammarSerializedFile")) {
                // load grammar from a serialized stanford parser file
                // following argument should be name of the parser file.
                op.grammarTypeFile = ParserOptions.FILE_TYPE.SERIALIZED;
                String[] files = ArgUtils.getFileNameFromArg(args, argIndex);
                if (files.length > 1) {
                    throw new RuntimeException("Can't have multiple grammar files.");
                }
                op.grammarFilePath = files[0];
                argIndex++;

            } else if (args[argIndex].equalsIgnoreCase("-saveOutputModelSerialized")) {
                // save output model as a serialized file
                op.outputTypeFile = ParserOptions.FILE_TYPE.SERIALIZED;
                String[] files = ArgUtils.getFileNameFromArg(args, argIndex);
                if (files.length > 1) {
                    throw new RuntimeException("Can't save multiple output files.");
                }
                op.outputFilePath = files[0];
                argIndex++;

            } else if (args[argIndex].equalsIgnoreCase("-saveOutputModelText")) {
                // set output model file as a text file
                op.outputTypeFile = ParserOptions.FILE_TYPE.TEXT;
                String[] files = ArgUtils.getFileNameFromArg(args, argIndex);
                if (files.length > 1) {
                    throw new RuntimeException("Can't save multiple output files.");
                }
                op.outputFilePath = files[0];
                argIndex++;

            } else if (args[argIndex].equalsIgnoreCase("-modelSerialized")) {
                // Serialized model file to be loaded, if testing
                op.modelTypeFile = ParserOptions.FILE_TYPE.SERIALIZED;
                String[] files = ArgUtils.getFileNameFromArg(args, argIndex);
                if (files.length > 1) {
                    throw new RuntimeException("Can't have multiple model files.");
                }
                op.modelFilePath = files[0];
                argIndex++;

            } else if (args[argIndex].equalsIgnoreCase("-modelText")) {
                // Text model file to be loaded if testing
                op.modelTypeFile = ParserOptions.FILE_TYPE.TEXT;
                String[] files = ArgUtils.getFileNameFromArg(args, argIndex);
                if (files.length > 1) {
                    throw new RuntimeException("Can't have multiple model files.");
                }
                op.modelFilePath = files[0];
                argIndex++;
            } else {
                // TODO: Additional arguments yet to be figured out

            }   // end arg parsing if statement
        }   // end while loop

        if (op.train) {
            // TODO:: Train weights and embeddings
        } else if (op.nbestRescore) {
            // TODO:: load model and re-rank files with options
        } else if (op.parse) {
            // TODO:: parse the files and generate the trees and return the score.

        } // end processing if statement
    }   // end of main


}
