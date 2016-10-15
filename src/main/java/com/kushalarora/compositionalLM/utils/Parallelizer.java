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

import static java.lang.Integer.max;
@Slf4j
public class Parallelizer {
    public Parallelizer(Options op, int blockSize) {
        this(op, blockSize, Executor.getInstance());
    }

    public Parallelizer(Options op, int blockSize, ExecutorService executor) {
        this.blockSize = blockSize;
        this.executor = executor;
    }


    private int blockSize;
    protected ExecutorService executor;

    public <D> List<Future<List<D>>> parallelizer(final int start, final int end,
                                                  final Function<Integer, D> parallizableFunc,
                                                  final int blockSize) {

        List<Callable<List<D>>> callables = new ArrayList<Callable<List<D>>>();
        int startIdx = start;
        while (startIdx < end) {
            int endIdx = (startIdx + blockSize < end) ? startIdx + blockSize : end;
            callables.add(new CallableWithInit<D>(startIdx, endIdx, parallizableFunc));
            startIdx = endIdx;
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
