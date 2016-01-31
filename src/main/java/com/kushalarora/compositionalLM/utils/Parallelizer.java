package com.kushalarora.compositionalLM.utils;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import com.google.common.base.Function;
import com.kushalarora.compositionalLM.options.Options;
import lombok.extern.slf4j.Slf4j;

/**
 * Created by arorak on 12/11/15.
 */
@Slf4j
public class Parallelizer {
    public Parallelizer(Options op, int blockSize) {
        this.blockSize = blockSize;
        executor = Executor.getInstance();
    }


    private int blockSize;
    protected ExecutorService executor;

    public <D> List<Future<List<D>>> parallelizer(final int start, final int end,
                                                  final Function<Integer, D> parallizableFunc,
                                                  final int blockSize) {
        int length = end - start;
        // Contains one extra block. Do something about it.
        int blockNum = length / blockSize;

        if (blockNum < 1) {
           /* log.error("blockNum is zero. " +
                    "Start: " + start +
                    " End: " + end  +
                    " blockSize: " + blockSize);*/
            blockNum = 1;
        }

        List<Callable<List<D>>> callables = new ArrayList<Callable<List<D>>>();
        for (int i = 0; i < blockNum; i++) {
            int startIdx = start + i * blockSize;
            int endIdx = (startIdx + blockSize < end) ? startIdx + blockSize : end;
            callables.add(new CallableWithInit<D>(startIdx, endIdx, parallizableFunc));
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

    public <D> List<Future<List<D>>> parallelizer(final int start, final int end, final Function<Integer, D> parallizableFunc) {
        return parallelizer(start, end, parallizableFunc, blockSize);
    }

    public class CallableWithInit<D> implements Callable<List<D>> {
        public int startIdx;
        public int endIndex;
        public Function<Integer, D> parallelizableFunc;
        public CallableWithInit(int startIndex, int endIndex, Function<Integer, D> parallelizableFunc) {
            this.startIdx = startIndex;
            this.endIndex = endIndex;
            this.parallelizableFunc = parallelizableFunc;
        }

        public List<D> call() throws Exception {
            List<D> dList = new ArrayList<D>();
            for (int j = startIdx; j < endIndex; j++) {
                dList.add(parallelizableFunc.apply(j));
            }
            return dList;
        }
    }
}
