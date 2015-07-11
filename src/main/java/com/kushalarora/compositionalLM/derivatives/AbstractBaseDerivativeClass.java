package com.kushalarora.compositionalLM.derivatives;

import com.kushalarora.compositionalLM.model.Model;

import java.io.Serializable;

/**
 * Created by karora on 6/30/15.
 */
public class AbstractBaseDerivativeClass implements Serializable{
    protected final Model model;

    public AbstractBaseDerivativeClass( Model model) {
        this.model = model;
    }
}
