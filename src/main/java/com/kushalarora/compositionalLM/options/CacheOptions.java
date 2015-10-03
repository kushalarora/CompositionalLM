package com.kushalarora.compositionalLM.options;

import com.kushalarora.compositionalLM.caching.CacheFactory;
import org.apache.commons.configuration.Configuration;

import java.io.Serializable;

/**
 * Created by arorak on 10/3/15.
 */
public class CacheOptions implements Serializable {
    public String cacheServer;
    public int cachePort;
    public CacheFactory.CacheType cacheType;
    public String mongodbCollection;
    public String mongodbDatabase;

    public CacheOptions(Configuration config) {
        cacheServer =
                config.getString("cacheServer", "localhost");

        cachePort =
                config.getInt("cachePort", 3030);

        cacheType =
                CacheFactory.CacheType.fromString(
                        config.getString("cacheType", "none"));

        mongodbCollection =
                config.getString("mongodbCollection", "test");

        mongodbDatabase =
                config.getString("mongoDB", "db");

    }
}
