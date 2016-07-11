package com.kushalarora.compositionalLM.utils;

import java.util.ArrayList;

import java.util.Random;
import javax.annotation.Nullable;

import org.apache.commons.configuration.ConfigurationException;
import org.junit.Assert;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import com.kushalarora.compositionalLM.options.Options;
import com.kushalarora.compositionalLM.utils.Parallelizer;

/**
 * Created by arorak on 12/12/15.
 */
public class ParallelizerTest
{
    private Parallelizer parallelizer;
    private static Options op;

    private class Accumulator implements Function<Integer, Void>
    {
        public Accumulator() {
            accumulator = 0;
            arrayInt = Lists.newArrayList(1,1,1,1,1,1,1,1,1);
        }

        @Nullable
        public Void apply(@Nullable Integer integer)
        {
            accumulator += arrayInt.get(integer);
            return null;
        }

        public int getAccumulator() {
            return accumulator;
        }

        ArrayList<Integer> arrayInt;
        public int accumulator;
    }
    @BeforeClass
    public static void setUpClass() throws ConfigurationException {
        op = new Options();
        op.trainOp.nThreads = 4;
    }

    @Before
    public void setUp() {
        parallelizer = new Parallelizer(op, 8);
    }

    @Test
    public void testSmallerThanABlock() {
        Accumulator acc = new Accumulator();
        parallelizer.parallelizer(0, 5, acc);
        Assert.assertEquals(5, acc.getAccumulator());
    }

    @Test
    public void testMultipleOfBlock() {
        Accumulator acc = new Accumulator();
        parallelizer.parallelizer(0, 8, acc);
        Assert.assertEquals(8, acc.getAccumulator());
    }

    @Test
    public void testSizeZero() {
        Accumulator acc = new Accumulator();
        parallelizer.parallelizer(0, 0, acc);
        Assert.assertEquals(0, acc.getAccumulator());
    }

    @Test
    public void testRandomLength()
    {
        Accumulator acc = new Accumulator();
        int size = new Random().nextInt();
        parallelizer.parallelizer(0, size, acc);
        Assert.assertEquals(size, acc.getAccumulator());
    }

}
