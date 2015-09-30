package com.kushalarora.compositionalLM.documentprocessor;

import com.kushalarora.compositionalLM.lang.Sentence;

import java.util.Iterator;

/**
 * Created by karora on 7/12/15.
 */
public abstract class DocumentProcessorWrapper<T> {
    protected int index;

    public DocumentProcessorWrapper() {
        index = 0;
    }

    public abstract Iterator<T> getIterator(String filename);


}
