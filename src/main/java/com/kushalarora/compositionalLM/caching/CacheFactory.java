package com.kushalarora.compositionalLM.caching;

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
        MONGO("mongo");

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

    private final Model model;

    public CacheFactory(Model model) {
        this.model = model;
    }

    public CacheWrapper getCache(Options op) throws IOException {
        switch (op.trainOp.cacheType) {
            case MEMCACHED:
                return new MemcachedWrapper<Sentence, IInsideOutsideScore>(op) {

                    @Override
                    public IInsideOutsideScore load(Sentence input) {
                        return model.getGrammar().computeScore(input);
                    }

                    @Override
                    public String getKeyString(Sentence input) {
                        StringBuilder sb = new StringBuilder();
                        for (Word word : input) {
                            sb.append(word.word()).append(":");
                        }
                        return sb.toString();
                    }
                };
            case NONE:
                return new CacheWrapper<Sentence, IInsideOutsideScore>() {

                    @Override
                    public IInsideOutsideScore load(Sentence input) {
                        return model.getGrammar().computeScore(input);
                    }

                    @Override
                    public void put(Sentence input, IInsideOutsideScore value) {
                        // do nothing
                    }

                    @Override
                    public IInsideOutsideScore getRoutine(Sentence input) {
                        return model.getGrammar().computeScore(input);
                    }

                    @Override
                    public void close() {
                        // do nothing
                    }

                    @Override
                    public String getKeyString(Sentence input) {
                        return null;
                    }
                };
            case EHCACHE:
                return new EhCacheWrapper<Sentence, IInsideOutsideScore>() {

                    @Override
                    public IInsideOutsideScore load(Sentence input) {
                        return model.getGrammar().computeScore(input);
                    }

                    @Override
                    public String getKeyString(Sentence input) {
                        StringBuilder sb = new StringBuilder();
                        for (Word word : input) {
                            sb.append(word.word()).append(":");
                        }
                        return sb.toString();
                    }
                };

            case REDIS:
                return new RedisCacheWrapper<Sentence, IInsideOutsideScore>(op) {

                    @Override
                    public IInsideOutsideScore load(Sentence input) {
                        return model.getGrammar().computeScore(input);
                    }

                    @Override
                    public String getKeyString(Sentence input) {
                        StringBuilder sb = new StringBuilder();
                        for (Word word : input) {
                            sb.append(word.word()).append(":");
                        }
                        return sb.toString();
                    }
                };

            case MONGO:
                return new MongoCacheWrapper<Sentence, IInsideOutsideScore>(op) {



                    @Override
                    public IInsideOutsideScore load(Sentence input) {
                        return (IInsideOutsideScore)model.getGrammar().computeScore(input);
                    }

                    @Override
                    public String getKeyString(Sentence input) {
                        StringBuilder sb = new StringBuilder();
                        for (Word word : input) {
                            sb.append(word.word()).append(":");
                        }
                        return sb.toString();
                    }
                };
            default:
                throw new RuntimeException("Invalid Cache Type: " + op.trainOp.cacheType);

        }
    }
}
