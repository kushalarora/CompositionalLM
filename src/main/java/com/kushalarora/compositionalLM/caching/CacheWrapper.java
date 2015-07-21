package com.kushalarora.compositionalLM.caching;

/**
 * Created by karora on 7/21/15.
 */
public abstract class CacheWrapper<K, V> {

    public abstract V load(K input);

    public abstract void put(K input, V value);

    public abstract V getRoutine(K input);

    public V get(K input) {
        V value = getRoutine(input);
        if (value == null) {
            value = load(input);
            put(input, value);
        }
        return value;
    }

    public abstract String getKeyString(K input);
}