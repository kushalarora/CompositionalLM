package com.kushalarora.compositionalLM.lang;

import com.kushalarora.compositionalLM.options.Options;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.process.DocumentPreprocessor;

import java.io.Reader;
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

    public DocumentProcessorWrapper getDocumentProcessor(Reader reader) {
        switch (op.grammarOp.grammarType) {
            case STANFORD_GRAMMAR:
                final DocumentPreprocessor processor =
                        new DocumentPreprocessor(reader, DocumentPreprocessor.DocType.Plain);

                if (!tokenizerFactory.op.grammarOp.grammarType
                        .equals(op.grammarOp.grammarType)) {
                    throw new RuntimeException("Inconsistent Tokenizer and document processor." +
                            " Document Processor: " + op.grammarOp.grammarType +
                            " Tokenizer: " + tokenizerFactory.op.grammarOp.grammarType);
                }
                processor.setTokenizerFactory(tokenizerFactory);

                return new DocumentProcessorWrapper() {
                    Iterator<List<HasWord>> it = processor.iterator();

                    public Iterator<List<Word>> iterator() {
                        return new Iterator<List<Word>>() {
                            public boolean hasNext() {
                                return it.hasNext();
                            }

                            public List<Word> next() {
                                List<Word> sentence =
                                        new ArrayList<Word>();
                                for (HasWord word : it.next()) {
                                    sentence.add((Word) word);
                                }
                                return sentence;
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
