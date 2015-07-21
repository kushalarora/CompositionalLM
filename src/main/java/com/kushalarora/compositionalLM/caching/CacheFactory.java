package com.kushalarora.compositionalLM.caching;

import com.kushalarora.compositionalLM.lang.IInsideOutsideScore;
import com.kushalarora.compositionalLM.lang.Word;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.options.Options;

import java.io.IOException;
import java.util.List;

/**
 * Created by karora on 7/21/15.
 */
public class CacheFactory {
    public enum CacheType {
        MEMCACHED("memcached"),
        NONE("none");
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
                return new MemcachedWrapper<List<Word>, IInsideOutsideScore>(op) {

                    @Override
                    public IInsideOutsideScore load(List<Word> input) {
                        return model.getGrammar().computeScore(input);
                    }

                    @Override
                    public String getKeyString(List<Word> input) {
                        return Integer.toString(input.hashCode());
                    }
                };
            case NONE:
                return new CacheWrapper<List<Word>, IInsideOutsideScore>() {

                    @Override
                    public IInsideOutsideScore load(List<Word> input) {
                        return model.getGrammar().computeScore(input);
                    }

                    @Override
                    public void put(List<Word> input, IInsideOutsideScore value) {
                        // do nothing
                    }

                    @Override
                    public IInsideOutsideScore getRoutine(List<Word> input) {
                        return model.getGrammar().computeScore(input);
                    }

                    @Override
                    public String getKeyString(List<Word> input) {
                        return null;
                    }
                };
            default:
                throw new RuntimeException("Invalid Cache Type: " + op.trainOp.cacheType);
        }
    }
}
