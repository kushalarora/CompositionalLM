package com.kushalarora.compositionalLM.derivatives;

import com.kushalarora.compositionalLM.model.CompositionalGrammar;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.model.Parameters;

/**
 * Created by karora on 6/30/15.
 */
public class AbstractBaseDerivativeClass {
    protected final Model model;

    public AbstractBaseDerivativeClass( Model model) {
        this.model = model;
    }
}
