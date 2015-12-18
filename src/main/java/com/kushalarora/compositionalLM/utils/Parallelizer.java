package com.kushalarora.compositionalLM.utils;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;

import com.google.common.base.Function;
import com.kushalarora.compositionalLM.options.Options;

/**
 * Created by arorak on 12/11/15.
 */
public class Parallelizer implements Serializable{
    public Parallelizer(Options op, int blockSize) {
        this.blockSize = blockSize;
        executor = Executor.getInstance();
    }


    private int blockSize;
    protected ExecutorService executor;

    public void parallelizer(final int start, final int end, final Function<Integer, Void> parallizableFunc)
    {
        int length = end - start;
        final int blockNum = length / blockSize + 1;

        List<Callable<Void>> callables = new ArrayList<Callable<Void>>();
        for (int i = 0; i < blockNum; i++)
        {
            callables.add(new Callable<Void>() {

                public Void call() throws Exception
                {
                    for (int j = blockStartIdx; j < blockStartIdx + blockSize && j < end; j++) {
                        parallizableFunc.apply(j);
                    }
                    return null;
                }

                public Callable<Void> init(int blockIndex) {
                    this.blockStartIdx = blockIndex;
                    return this;
                }

                private int blockStartIdx;
            }.init(start + i * blockSize));
        }

        try
        {
            executor.invokeAll(callables);
        }
        catch (InterruptedException e)
        {
            throw new RuntimeException(e);
        }
    }
}
