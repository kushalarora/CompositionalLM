package com.kushalarora.compositionalLM.documentprocessor;

import com.kushalarora.compositionalLM.lang.Sentence;
import com.kushalarora.compositionalLM.lang.TokenizerFactory;
import com.kushalarora.compositionalLM.lang.Word;
import com.kushalarora.compositionalLM.options.Options;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.process.DocumentPreprocessor;
import org.apache.commons.collections.Predicate;
import org.apache.commons.collections.iterators.FilterIterator;

import java.io.Reader;
import java.util.Iterator;
import java.util.List;

/**
 * Created by karora on 7/29/15.
 */
public class StanfordDocumentProcessor extends DocumentProcessorWrapper<Sentence> {
    final TokenizerFactory tokenizerFactory;
    final Options op;

    public StanfordDocumentProcessor(Options op, TokenizerFactory tokenizerFactory) {
        this.op = op;
        this.tokenizerFactory = tokenizerFactory;

        if (!op.grammarOp.grammarType.equals(tokenizerFactory.getOp().grammarOp.grammarType)) {
            throw new RuntimeException("GrammarType for tokenizer(" +
                    tokenizerFactory.getOp().grammarOp.grammarType + ") " +
                    "doesn't match " +
                    "document processor(" +
                    op.grammarOp.grammarType + ")");
        }
    }

    /*
        Wrapper around the Stanford Iterator.
        Returns Sentence instead of List<HasWord>

        TODO: Write tests
     */
    @Override
    public Iterator<Sentence> getIterator(String filename) {
        DocumentPreprocessor processor = new DocumentPreprocessor(filename,
                DocumentPreprocessor.DocType.Plain);

        processor.setTokenizerFactory(tokenizerFactory);


        if (op.grammarOp.newLineDelimiter) {
            processor.setSentenceDelimiter("\n");
        }

        final Iterator<List<HasWord>> it = processor.iterator();

        return new FilterIterator(new Iterator<Sentence>() {

            public boolean hasNext() {

                return it.hasNext();
            }

            public Sentence next() {
                return transform(it.next());
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
        }, new Predicate() {
            public boolean evaluate(Object o) {
                Sentence sent = (Sentence) o;
                return (sent.size() <= op.grammarOp.maxLength);
            }
        });
    }
}
