package com.kushalarora.compositionalLM.lang;

/**
 * Created by karora on 7/12/15.
 */
public abstract class DocumentProcessorWrapper implements Iterable<Sentence> {
    protected int index;

    public DocumentProcessorWrapper() {
        index = 0;
    }


}
