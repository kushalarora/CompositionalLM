package com.kushalarora.compositionalLM.caching;

import com.kushalarora.compositionalLM.options.Options;
import net.spy.memcached.AddrUtil;
import net.spy.memcached.MemcachedClient;

import java.io.IOException;

/**
 * Created by karora on 7/21/15.
 */
public abstract class MemcachedWrapper<K, V> extends CacheWrapper<K, V> {

    MemcachedClient memcachedClient;

    // never expire
    public static int EXPIRY = 0;

    public MemcachedWrapper(Options op) throws IOException {
        memcachedClient = new MemcachedClient(
                AddrUtil.getAddresses(op.cacheOp.cacheServer + ":" + op.cacheOp.cachePort));
    }

    @Override
    public synchronized V getRoutine(K input) {
        return (V) memcachedClient.get(getKeyString(input));
    }

    @Override
    public synchronized void put(K input, V value) {
        memcachedClient.set(getKeyString(input), EXPIRY, value);
    }


    @Override
    public void close() {
        memcachedClient.shutdown();
    }


    public void clear() {
        memcachedClient.flush();
    }
}
