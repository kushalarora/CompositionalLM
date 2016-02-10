package com.kushalarora.compositionalLM.caching;

import net.sf.ehcache.CacheManager;
import net.sf.ehcache.Ehcache;
import net.sf.ehcache.Element;

/**
 * Created by karora on 7/25/15.
 */
public abstract class EhCacheWrapper<K, V> extends CacheWrapper<K, V> {

    public static String CACHE_NAME = "cache";

    @Override
    public void put(K input, V value) {
        getCache().put(new Element(getKeyString(input), value));
    }

    @Override
    public V getRoutine(K input) {
        Ehcache cache = getCache();
        Element element = null;
        if (cache != null) {
            element = cache.get(getKeyString(input));
        }

        if (element == null) {
            return null;
        }
        return (V) element.getObjectValue();
    }


    public static CacheManager cacheMgr = null;

    private static Ehcache getCache() {
        if (cacheMgr == null) {
            // We could use an environment or a VM variable
            cacheMgr = CacheManager.create("ehcache.xml");
        }

        Ehcache cache = null;
        if (cacheMgr != null) {
            //cache = cacheMgr.addCacheIfAbsent(name);
            cache = cacheMgr.getEhcache(CACHE_NAME);
        }

        return cache;
    }

    public void clear() {
        getCache().removeAll();
    }

    @Override
    public void close() {
        cacheMgr.shutdown();
    }
}
