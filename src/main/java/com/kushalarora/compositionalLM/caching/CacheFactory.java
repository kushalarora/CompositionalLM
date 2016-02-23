package com.kushalarora.compositionalLM.caching;

import com.google.common.base.Function;
import com.kushalarora.compositionalLM.lang.IGrammar;
import com.kushalarora.compositionalLM.lang.IInsideOutsideScore;
import com.kushalarora.compositionalLM.lang.Sentence;
import com.kushalarora.compositionalLM.lang.Word;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.options.Options;

import java.io.IOException;

/**
 * Created by karora on 7/21/15.
 */
public class CacheFactory {
    public enum CacheType {
        MEMCACHED("memcached"),
        NONE("none"),
        EHCACHE("ehcache"),
        REDIS("redis"),
        MONGO("mongo"),
        GUAVA("guava");

        private String text;

        CacheType(String text) {
            this.text = text;
        }

        public String getText() {
            return this.text;
        }

        public static CacheType fromString(String text) {
            if (text != null) {
                for (CacheType b : CacheType.values()) {
                    if (text.equalsIgnoreCase(b.text)) {
                        return b;
                    }
                }
            }
            return null;
        }
    }


    public CacheFactory() {

    }

    public static  <K,V> CacheWrapper getCache(Options op, final Function<K, V> loadFunc) throws IOException {
        switch (op.cacheOp.cacheType) {
            case MEMCACHED:
                return new MemcachedWrapper<K, V>(op) {

                    @Override
                    public V loadValue(K input) {
                        return loadFunc.apply(input);
                    }

                    @Override
                    public String getKeyString(K input) {
                        return input.toString().replace(" ", ":");
                    }
                };
            case NONE:
                return new CacheWrapper<K, V>() {

                    @Override
                    public V loadValue(K input) {
                        return loadFunc.apply(input);
                    }

                    @Override
                    public void put(K input, V value) {
                        // do nothing
                    }

                    @Override
                    public V getRoutine(K input) {
                        return loadFunc.apply(input);
                    }

                    @Override
                    public void close() {
                        // do nothing
                    }

                    @Override
                    public void clear() {}

                    @Override
                    public String getKeyString(K input) {
                        return null;
                    }
                };
            case EHCACHE:
                return new EhCacheWrapper<K, V>() {

                    @Override
                    public V loadValue(K input) {
                        return loadFunc.apply(input);
                    }

                    @Override
                    public String getKeyString(K input) {
                        return input.toString();
                    }
                };

            case REDIS:
                return new RedisCacheWrapper<K, V>(op) {
                    @Override
                    public V loadValue(K input) {
                        return loadFunc.apply(input);
                    }

                    @Override
                    public String getKeyString(K input) {
                        return input.toString();
                    }
                };

            case MONGO:
                return new MongoCacheWrapper<K, V>(op) {

                    @Override
                    public V loadValue(K input) {
                        return loadFunc.apply(input);
                    }

                    @Override
                    public String getKeyString(K input) {
                        return input.toString();
                    }
                };
            case GUAVA:
                return new GuavaCacheWrapper<K,V>(op) {
                    @Override
                    public V loadValue(K input) {
                        return loadFunc.apply(input);
                    }

                    @Override
                    public V getRoutine(K input) {
                        return null;
                    }

                    @Override
                    public String getKeyString(K input) {
                        return input.toString();
                    }
                };
            default:
                throw new RuntimeException("Invalid Cache Type: " + op.cacheOp.cacheType);

        }
    }
}
