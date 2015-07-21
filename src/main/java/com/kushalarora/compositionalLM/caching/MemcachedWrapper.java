package com.kushalarora.compositionalLM.caching;

import com.kushalarora.compositionalLM.options.Options;
import net.spy.memcached.AddrUtil;
import net.spy.memcached.MemcachedClient;

import java.io.IOException;
import java.net.InetSocketAddress;

/**
 * Created by karora on 7/21/15.
 */
public abstract class MemcachedWrapper<K, V> extends CacheWrapper<K, V> {

    MemcachedClient memcachedClient;

    public static int EXPIRY = 3600;

    public MemcachedWrapper(Options op) throws IOException {
        memcachedClient =new MemcachedClient(
                AddrUtil.getAddresses("localhost:3030"));
        ;
    }

    @Override
    public void put(K input, V value) {
        memcachedClient.set(getKeyString(input), EXPIRY, value);
    }

    @Override
    public V getRoutine(K input) {
        return (V)memcachedClient.get(getKeyString(input));
    }
}
