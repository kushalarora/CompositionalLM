package com.kushalarora.compositionalLM.utils;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import com.google.common.base.Function;
import com.kushalarora.compositionalLM.options.Options;

/**
 * Created by arorak on 12/11/15.
 */
public class Parallelizer {
    public Parallelizer(Options op, int blockSize) {
        this.blockSize = blockSize;
        executor = Executor.getInstance();
    }


    private int blockSize;
    protected ExecutorService executor;

    public <D> List<Future<List<D>>> parallelizer(final int start, final int end, final Function<Integer, D> parallizableFunc) {
        int length = end - start;
        final int blockNum = length / blockSize + 1;

        List<Callable<List<D>>> callables = new ArrayList<Callable<List<D>>>();
        for (int i = 0; i < blockNum; i++) {
            callables.add(new Callable<List<D>>() {

                public List<D> call() throws Exception {
                    List<D> dList = new ArrayList<D>();
                    for (int j = blockStartIdx; j < blockStartIdx + blockSize && j < end; j++) {
                        dList.add(parallizableFunc.apply(j));
                    }
                    return dList;
                }

                public Callable<List<D>> init(int blockIndex) {
                    this.blockStartIdx = blockIndex;
                    return this;
                }

                private int blockStartIdx;
            }.init(start + i * blockSize));
        }

        try
        {
            return executor.invokeAll(callables);
        }
        catch (InterruptedException e)
        {
            throw new RuntimeException(e);
        }
    }
}
