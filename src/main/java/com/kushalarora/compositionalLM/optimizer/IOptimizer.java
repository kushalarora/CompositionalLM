package com.kushalarora.compositionalLM.optimizer;

/**
 * Created by karora on 7/7/15.
 */
public interface IOptimizer {

    boolean done();

    void calcDerivativeAndUpdate();
}
