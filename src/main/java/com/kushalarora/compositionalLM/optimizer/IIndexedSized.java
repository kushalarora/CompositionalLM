package com.kushalarora.compositionalLM.optimizer;

/**
 * Created by arorak on 10/22/15.
 */
public interface IIndexedSized extends IIndexed {
    public int getSize();
    public IIndexed get(int i);
}
