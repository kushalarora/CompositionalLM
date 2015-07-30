package com.kushalarora.compositionalLM.documentprocessor;

import com.kushalarora.compositionalLM.documentprocessor.DocumentProcessorWrapper;
import com.kushalarora.compositionalLM.lang.Sentence;
import com.kushalarora.compositionalLM.lang.TokenizerFactory;
import com.kushalarora.compositionalLM.lang.Word;
import com.kushalarora.compositionalLM.options.Options;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.process.DocumentPreprocessor;

import java.io.Reader;
import java.util.Iterator;
import java.util.List;

/**
 * Created by karora on 7/29/15.
 */
public class StanfordDocumentProcessor extends DocumentProcessorWrapper {
    final DocumentPreprocessor processor;

    public StanfordDocumentProcessor(Options op, String filename, TokenizerFactory tokenizerFactory) {
        processor = new DocumentPreprocessor(filename,
                DocumentPreprocessor.DocType.Plain);

        if (!op.grammarOp.grammarType.equals(tokenizerFactory.getOp().grammarOp.grammarType)) {
            throw new RuntimeException("GrammarType for tokenizer(" +
                    tokenizerFactory.getOp().grammarOp.grammarType + ") " +
                    "doesn't match " +
                    "document processor(" +
                    op.grammarOp.grammarType + ")");
        }

        processor.setTokenizerFactory(tokenizerFactory);


        if (op.grammarOp.newLineDelimiter) {
            processor.setSentenceDelimiter("\n");
        }

    }

    public StanfordDocumentProcessor(Options op, Reader reader, TokenizerFactory tokenizerFactory) {
        processor = new DocumentPreprocessor(reader,
                DocumentPreprocessor.DocType.Plain);

        processor.setTokenizerFactory(tokenizerFactory);

        if (op.grammarOp.newLineDelimiter) {
            processor.setSentenceDelimiter("\n");
        }

    }

    public Iterator<Sentence> iterator() {
        final Iterator<List<HasWord>> it = processor.iterator();

        return new Iterator<Sentence>() {

            public boolean hasNext() {

                return it.hasNext();
            }

            public Sentence next() {
                Sentence sentence =
                        new Sentence(index);

                for (HasWord word : it.next()) {
                    sentence.add((Word) word);
                }

                index++;
                return sentence;
            }

            public void remove() {
                it.remove();
            }
        };
    }
}
