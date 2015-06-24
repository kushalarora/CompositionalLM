package com.kushalarora.compositionalLM.model;

import com.kushalarora.compositionalLM.lang.IGrammar;

/**
 * Created by karora on 6/21/15.
 */
public class CompositionalGrammar {
    private  Model model;
    private IGrammar grammar;

    public CompositionalGrammar(IGrammar grammar, Model model) {
        this.model = model;
        this.grammar = grammar;
    }

    public void train() {

    }



}
