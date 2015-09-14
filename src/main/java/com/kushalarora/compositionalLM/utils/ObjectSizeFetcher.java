package com.kushalarora.compositionalLM.utils;

import org.github.jamm.MemoryMeter;

import java.lang.instrument.Instrumentation;

/**
 * Created by arorak on 9/13/15.
 */
public class ObjectSizeFetcher {
    private static MemoryMeter meter;
    private static ObjectSizeFetcher fetcher;

    private  void init() {
        meter = new MemoryMeter();
    }


    public static double getSize(Object obj) {
        if (meter == null) {
            meter = new MemoryMeter();
        }

        return meter.measureDeep(obj)/(1024.0 * 1024.0);
    }
}
