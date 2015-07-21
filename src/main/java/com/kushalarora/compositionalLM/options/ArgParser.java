package com.kushalarora.compositionalLM.options;

import com.kushalarora.compositionalLM.lang.GrammarFactory;
import com.kushalarora.compositionalLM.utils.ArgUtils;
import lombok.extern.slf4j.Slf4j;

/**
 * Created by karora on 7/12/15.
 */

@Slf4j
public class ArgParser {
    public static Options parseArgs(String[] args) {
        int argIndex = 0;
        Options op = new Options();
        if (args.length < 1) {
            log.error("TODO: Write Error message about too few arguments");
        }

        while (argIndex < args.length) {
            if (args[argIndex].equalsIgnoreCase("-train")) {
                // Must be followed by a training file
                op.train = true;
                op.trainOp.trainFiles = ArgUtils.getStringFromArg(args, argIndex);
                argIndex += ArgUtils.numSubArgs(args, argIndex);

            } else if (args[argIndex].equalsIgnoreCase("-nbest")) {
                // Must be followed by a text file with n best list generated from
                // lattice decoder to be reranked
                op.nbestRescore = true;
                op.testOp.nbestFiles = ArgUtils.getStringFromArg(args, argIndex);
                argIndex += ArgUtils.numSubArgs(args, argIndex);

            } else if (args[argIndex].equalsIgnoreCase("-validate")) {
                // Ignored if only testing.
                // If training must be followed by validation File
                op.trainOp.validate = true;
                op.trainOp.validationFiles = ArgUtils.getStringFromArg(args, argIndex);
                argIndex += ArgUtils.numSubArgs(args, argIndex);

            } else if (args[argIndex].equalsIgnoreCase("-parse")) {
                // Followed by file to be parsed.
                op.parse = true;
                op.testOp.parseFiles = ArgUtils.getStringFromArg(args, argIndex);
                argIndex += ArgUtils.numSubArgs(args, argIndex);

            } else if (args[argIndex].equalsIgnoreCase("-grammarFile")) {
                // load grammar from a serialized stanford parser file
                // following argument should be name of the parser file.
                String[] files = ArgUtils.getStringFromArg(args, argIndex);
                if (files.length > 1) {
                    throw new RuntimeException("Can't have multiple grammar files.");
                }
                op.grammarOp.filename = files[0];
                argIndex++;

            } else if (args[argIndex].equalsIgnoreCase("-saveOutputModelSerialized")) {
                // save output model as a serialized file
                op.modelOp.outType = Options.FILE_TYPE.SERIALIZED;
                String[] files = ArgUtils.getStringFromArg(args, argIndex);
                if (files.length > 1) {
                    throw new RuntimeException("Can't save multiple output files.");
                }
                op.modelOp.outFilename = files[0];
                argIndex++;

            } else if (args[argIndex].equalsIgnoreCase("-saveOutputModelText")) {
                // set output model file as a text file
                op.modelOp.outType = Options.FILE_TYPE.TEXT;
                String[] files = ArgUtils.getStringFromArg(args, argIndex);
                if (files.length > 1) {
                    throw new RuntimeException("Can't save multiple output files.");
                }
                op.modelOp.outFilename = files[0];
                argIndex++;

            } else if (args[argIndex].equalsIgnoreCase("-modelSerialized")) {
                // Serialized model file to be loaded, if testing
                op.modelOp.inType = Options.FILE_TYPE.SERIALIZED;
                String[] files = ArgUtils.getStringFromArg(args, argIndex);
                if (files.length > 1) {
                    throw new RuntimeException("Can't have multiple model files.");
                }
                op.modelOp.outFilename = files[0];
                argIndex++;

            } else if (args[argIndex].equalsIgnoreCase("-modelText")) {
                // Text model file to be loaded if testing
                op.modelOp.inType = Options.FILE_TYPE.TEXT;
                String[] files = ArgUtils.getStringFromArg(args, argIndex);
                if (files.length > 1) {
                    throw new RuntimeException("Can't have multiple model files.");
                }
                op.modelOp.inFilename = files[0];
                argIndex++;

            } else if (args[argIndex].equalsIgnoreCase("-grammarType")) {
                String[] grammars = ArgUtils.getStringFromArg(args, argIndex);
                if (grammars.length > 1) {
                    throw new RuntimeException("You can specify only one grammarType of grammar");
                }
                op.grammarOp.grammarType = GrammarFactory.GrammarType.fromString(grammars[0]);
                argIndex++;

            } else if (args[argIndex].equalsIgnoreCase("-dimension")) {
                String[] dimensions = ArgUtils.getStringFromArg(args, argIndex);
                if (dimensions.length > 1) {
                    throw new RuntimeException("You can only specify one dimension value");
                }
                op.modelOp.dimensions = Integer.parseInt(dimensions[0]);
                argIndex++;
            } else if (args[argIndex].equalsIgnoreCase("-lowercase")) {
                op.grammarOp.lowerCase = true;
            } else if (args[argIndex].equalsIgnoreCase("-parallel")) {
                op.trainOp.parallel = true;
                op.trainOp.nThreads = Runtime.getRuntime().availableProcessors();
            } else if (args[argIndex].equalsIgnoreCase("-parallel")) {
                String[] numThreads = ArgUtils.getStringFromArg(args, argIndex);
                if (numThreads.length > 1) {
                    throw new RuntimeException("You can only specify one numThread value");
                }
                op.trainOp.parallel =  true;
                op.trainOp.nThreads = Integer.parseInt(numThreads[0]);
                argIndex++;
            } else {

            }   // end arg parsing if statement
            argIndex++;
        }   // end while loop
        return op;
    }
}
