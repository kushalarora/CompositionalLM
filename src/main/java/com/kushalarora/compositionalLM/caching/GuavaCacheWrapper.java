package com.kushalarora.compositionalLM.caching;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.kushalarora.compositionalLM.options.Options;
import lombok.SneakyThrows;

import java.io.IOException;
import java.util.concurrent.Callable;

/**
 * Created by arorak on 2/22/16.
 */
public abstract class GuavaCacheWrapper<K, V> extends CacheWrapper<K, V> {

    LoadingCache<K, V> cache;

    public GuavaCacheWrapper(Options op) throws IOException {
        cache = CacheBuilder.newBuilder()
                .maximumSize(200000)
                .build(new CacheLoader<K, V>() {
                            public V load(K key) {
                                return loadValue(key);
                            }
                        });
    }

    @Override
    public void put(K input, V value) {
        cache.put(input, value);
    }

    @Override
    @SneakyThrows
    public V get(final K input) {
        return cache.get(input, new Callable<V>() {
            public  V call() throws Exception {
                return loadValue(input);
            }
        });
    }

    @Override
    public void close() {
        cache.invalidateAll();
    }

    @Override
    public void clear() throws Exception {

    }
}
