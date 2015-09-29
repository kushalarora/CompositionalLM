package com.kushalarora.compositionalLM.documentprocessor;

import com.kushalarora.compositionalLM.lang.Sentence;
import com.kushalarora.compositionalLM.lang.TokenizerFactory;
import com.kushalarora.compositionalLM.options.Options;
import edu.stanford.nlp.io.IOUtils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Iterator;
import java.util.regex.Pattern;

/**
 * Created by karora on 7/29/15.
 */
public class MSSCProcessor extends DocumentProcessorWrapper {

    Pattern END_HEADER_PATTERN = Pattern.compile("[\\*]?END[\\*]?.*[\\*]?END[\\*]?");

    StanfordDocumentProcessor stanfordProcessor;

    public MSSCProcessor(Options op, String filename, TokenizerFactory tokenizerFactory) throws IOException {

        BufferedReader reader = IOUtils.readerFromString(filename);

        Path path = Paths.get(filename);
        String tmpFileName = "/tmp/" + path.getFileName();
        BufferedWriter writer = Files.newBufferedWriter(Paths.get(tmpFileName),
                Charset.forName("UTF-8"));

        boolean matched = false;
        String line;
        while ((line = reader.readLine()) != null && !matched) {
            if (END_HEADER_PATTERN.matcher(line).find()) {
                matched = true;
                break;
            }
        }

        if (!matched) {
            throw new RuntimeException("Header not found in file: " + filename);
        }

        while ((line = reader.readLine()) != null) {
            line = line.replace("\"", "");
            writer.write(line + "\n");
        }

        writer.close();
        reader.close();

        stanfordProcessor = new StanfordDocumentProcessor(op, tmpFileName, tokenizerFactory);
    }

    public Iterator<Sentence> iterator() {
        return stanfordProcessor.iterator();
    }
}
