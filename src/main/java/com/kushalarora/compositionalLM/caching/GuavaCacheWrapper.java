package com.kushalarora.compositionalLM.caching;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;

import com.kushalarora.compositionalLM.options.Options;
import lombok.SneakyThrows;

import java.io.IOException;

/**
 * Created by arorak on 2/22/16.
 */
public abstract class GuavaCacheWrapper<K, V> extends CacheWrapper<K, V> {

    Cache<K, V> cache;

    public GuavaCacheWrapper(Options op) throws IOException {
        cache = Caffeine.newBuilder()
                .maximumSize(10000)
                .build();
    }

    @Override
    public void put(K input, V value) {
        cache.put(input, value);
    }

    @Override
    @SneakyThrows
    public V getRoutine(final K input) {
        return cache.getIfPresent(input);
    }

    @Override
    public void close() {
        cache.invalidateAll();
    }

    @Override
    public void clear() throws Exception {
        cache.invalidateAll();
    }
}
