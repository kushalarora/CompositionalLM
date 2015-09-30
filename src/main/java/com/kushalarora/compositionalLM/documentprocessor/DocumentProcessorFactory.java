package com.kushalarora.compositionalLM.documentprocessor;

import com.kushalarora.compositionalLM.lang.TokenizerFactory;
import com.kushalarora.compositionalLM.options.Options;

import java.io.IOException;

/**
 * Created by karora on 7/12/15.
 */
public class DocumentProcessorFactory {

    public enum DocumentProcessorType {
        STANFORD_PLAIN("stanford_plain"),
        BERKELEY_PLAIN("berkeley_plain"),
        MS_SENTENCE_CHALLENGE("ms_sents");

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

    public DocumentProcessorWrapper getDocumentProcessor() throws IOException {
        switch (op.inputOp.processorType) {
            case STANFORD_PLAIN:
                return new StanfordDocumentProcessor(op, tokenizerFactory);

            case MS_SENTENCE_CHALLENGE:
                return new MSSCProcessor(op, tokenizerFactory);

            case BERKELEY_PLAIN:

            default:
                throw new RuntimeException("Invalid Grammar Type: " + op.grammarOp.grammarType);
        }
    }
}
