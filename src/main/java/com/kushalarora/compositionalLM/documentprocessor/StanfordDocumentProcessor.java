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
    final Options op;

    public StanfordDocumentProcessor(Options op, String filename, TokenizerFactory tokenizerFactory) {
        processor = new DocumentPreprocessor(filename,
                DocumentPreprocessor.DocType.Plain);

        this.op = op;

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

        this.op = op;

        processor.setTokenizerFactory(tokenizerFactory);

        if (op.grammarOp.newLineDelimiter) {
            processor.setSentenceDelimiter("\n");
        }

    }

    /*
        Wrapper around the Stanford Iterator.
        Returns Sentence instead of List<HasWord>

        TODO: Write tests
     */
    public Iterator<Sentence> iterator() {
        final Iterator<List<HasWord>> it = processor.iterator();


        return new Iterator<Sentence>() {
            List<HasWord> sentence;

            public boolean hasNext() {

                boolean hasNext;
                while ((hasNext = it.hasNext()) && // if next element present
                        (sentence = it.next()).size() > op.grammarOp.maxLength); // and is greater than maxLength
                // keep ignoring

                return hasNext;
            }

            public Sentence next() {
                return transform(sentence);
            }

            private Sentence transform(List<HasWord> sent) {
                Sentence sentence =
                        new Sentence(index);

                for (HasWord word : sent) {
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
