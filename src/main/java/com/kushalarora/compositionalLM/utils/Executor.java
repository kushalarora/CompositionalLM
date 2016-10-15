package com.kushalarora.compositionalLM.utils;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Created by arorak on 12/17/15.
 */
public class Executor {

    public static ExecutorService getInstance() {
        if (executorService == null) {
            executorService = Executors.newCachedThreadPool();
        }
        return executorService;
    }

    private static ExecutorService executorService;
}
