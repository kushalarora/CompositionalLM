package com.kushalarora.compositionalLM.utils;

import java.io.Serializable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import com.google.common.base.Function;
import com.kushalarora.compositionalLM.options.Options;

/**
 * Created by arorak on 12/11/15.
 */
public class Parallelizer implements Serializable{
    public Parallelizer(Options op) {
        this.blockSize = op.trainOp.blockSize;
        executor = Executors.newFixedThreadPool((int)Math.floor(Math.sqrt(2 * op.trainOp.nThreads)));
    }


    private int blockSize;
    protected ExecutorService executor;

    public void parallelizer(final int start, final int end, final Function<Integer, Void> parallizableFunc)
    {
        int length = end - start;
        final int blockNum = length / blockSize + 1;

        for (int i = 0; i < blockNum; i++)
        {
            executor.submit(new Runnable() {

                public void run() {
                    for (int j = blockStartIdx; j < blockStartIdx + blockSize && j < end; j++) {
                        parallizableFunc.apply(j);
                    }
                }

                public Runnable init(int blockIndex) {
                    this.blockStartIdx = blockIndex;
                    return this;
                }

                private int blockStartIdx;
            }.init(start + i * blockSize));
        }
    }
}
