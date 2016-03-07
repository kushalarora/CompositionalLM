package com.kushalarora.compositionalLM.options;

import com.kushalarora.compositionalLM.documentprocessor.DocumentProcessorFactory;
import lombok.ToString;
import org.apache.commons.configuration.Configuration;

import java.io.Serializable;


/**
 * Created by karora on 7/29/15.
 */
@ToString
public class InputOptions implements Serializable {
    public int DEFAULT_MAX_LENGTH = 30;

    public DocumentProcessorFactory.DocumentProcessorType processorType;
    public int maxLength;
    public double errPct;

    public InputOptions(Configuration config) {
        processorType = DocumentProcessorFactory
                .DocumentProcessorType.fromString(
                        config.getString("processorType", "stanford"));

        maxLength = config.getInt("maxLength", DEFAULT_MAX_LENGTH);

        errPct = config.getDouble("errPct", 25d);
    }
}
