package com.kushalarora.compositionalLM.documentprocessor;

import com.kushalarora.compositionalLM.lang.TokenizerFactory;
import com.kushalarora.compositionalLM.options.Options;

import java.io.IOException;

/**
 * Created by karora on 7/12/15.
 */
public class DocumentProcessorFactory {

    public enum DocumentProcessorType {
        STANFORD_PLAIN("stanford"),
        BERKELEY_PLAIN("berkeley"),
        MS_SENTENCE_CHALLENGE("mssc");

        private String text;

        DocumentProcessorType(String text) {
            this.text = text;
        }

        public String getText() {
            return this.text;
        }

        public static DocumentProcessorType fromString(String text) {
            if (text != null) {
                for (DocumentProcessorType b : DocumentProcessorType.values()) {
                    if (text.equalsIgnoreCase(b.text)) {
                        return b;
                    }
                }
            }
            return null;
        }
    }

    private Options op;
    private TokenizerFactory tokenizerFactory;

    public DocumentProcessorFactory(Options op, TokenizerFactory tokenizerFactory) {
        this.op = op;
        this.tokenizerFactory = tokenizerFactory;
    }

    public DocumentProcessorWrapper getDocumentProcessor(String filename) throws IOException {
        switch (op.inputOp.processorType) {
            case STANFORD_PLAIN:
                return new StanfordDocumentProcessor(op, filename, tokenizerFactory);

            case MS_SENTENCE_CHALLENGE:
                return new MSSCProcessor(op, filename, tokenizerFactory);

            case BERKELEY_PLAIN:

            default:
                throw new RuntimeException("Invalid Grammar Type: " + op.grammarOp.grammarType);
        }
    }
}
