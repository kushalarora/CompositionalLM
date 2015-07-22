package com.kushalarora.compositionalLM.lang;

import com.kushalarora.compositionalLM.options.Options;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.process.DocumentPreprocessor;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Created by karora on 7/12/15.
 */
public class DocumentProcessorFactory {
    private Options op;
    private TokenizerFactory tokenizerFactory;

    public DocumentProcessorFactory(Options op, TokenizerFactory tokenizerFactory) {
        this.op = op;
        this.tokenizerFactory = tokenizerFactory;
    }

    public DocumentProcessorWrapper getDocumentProcessor(String filename) {
        switch (op.grammarOp.grammarType) {
            case STANFORD_GRAMMAR:
                final DocumentPreprocessor processor =
                        new DocumentPreprocessor(filename,
                                DocumentPreprocessor.DocType.Plain);

                if (!tokenizerFactory.op.grammarOp.grammarType
                        .equals(op.grammarOp.grammarType)) {
                    throw new RuntimeException("Inconsistent Tokenizer and document processor." +
                            " Document Processor: " + op.grammarOp.grammarType +
                            " Tokenizer: " + tokenizerFactory.op.grammarOp.grammarType);
                }
                processor.setTokenizerFactory(tokenizerFactory);
                // TODO:: Don't hardcode. Figure out how to add more than one.
                processor.setSentenceDelimiter("\n");

                final Iterator<List<HasWord>> it = processor.iterator();

                return new DocumentProcessorWrapper() {
                    public Iterator<Sentence> iterator() {
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
                };

            case BERKELEY_GRAMMAR:

            default:
                throw new RuntimeException("Invalid Grammar Type: " + op.grammarOp.grammarType);
        }
    }
}
