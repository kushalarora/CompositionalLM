package com.kushalarora.compositionalLM.lang;

import com.kushalarora.compositionalLM.optimizer.IIndexed;
import com.kushalarora.compositionalLM.optimizer.IIndexedSized;

import java.util.ArrayList;

/**
 * Created by karora on 7/22/15.
 */
public class Sentence extends ArrayList<Word> implements IIndexedSized
{
    private int index;

    public Sentence(int index) {
        this.index = index;
    }

    @Override
    public String toString() {
        String[] strings = new String[size()];
        int idx = 0;
        for (Word word : this) {
            strings[idx] = word.word();
            idx++;
        }
        return String.join(" ", strings);
    }

    public int getIndex() {
        return index;
    }

    public int getSize() {
        return size();
    }

}
