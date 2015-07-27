package com.kushalarora.compositionalLM.caching;

import lombok.extern.slf4j.Slf4j;

/**
 * Created by karora on 7/21/15.
 */

@Slf4j
public abstract class CacheWrapper<K, V> {

    public abstract V load(K input);

    public abstract void put(K input, V value);

    public abstract V getRoutine(K input);

    public V get(K input) {
        V value = getRoutine(input);
        if (value == null) {
            value = load(input);
            log.warn("value not found for input: {}", getKeyString(input));
            synchronized (this) {
                put(input, value);
            }
        }
        return value;
    }

    public abstract void close();

    public abstract String getKeyString(K input);
}