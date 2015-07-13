package com.kushalarora.compositionalLM.options;

import java.io.Serializable;

/**
 * Created by karora on 6/14/15.
 */
public class TrainOptions implements Serializable {
    public String[] trainFiles = null;
    public boolean train = false;

    public boolean validate = false;
    public String[] validationFiles = null;
}
