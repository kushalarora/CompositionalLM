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
    public DocumentProcessorFactory.DocumentProcessorType processorType;

    public InputOptions(Configuration config) {
        processorType = DocumentProcessorFactory
                .DocumentProcessorType.fromString(
                        config.getString("processorType", "stanford_plain"));
    }
}
