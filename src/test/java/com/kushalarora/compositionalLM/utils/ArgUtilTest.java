package com.kushalarora.compositionalLM.utils;

import com.kushalarora.compositionalLM.utils.ArgUtils;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Created by karora on 7/12/15.
 */
public class ArgUtilTest {

    public static String[] args;

    @BeforeClass
    public static void setUpClass() {
        args = new String[]{
                "-arg1Header", "arg11", "arg12", "arg13",
                "-arg2Header", "arg21", "arg22"};
    }


    @Test
    public void testGetStringsFromArgs() {
        String[] arg = ArgUtils.getStringFromArg(args, 0);
        assertEquals(3, arg.length);
        assertArrayEquals(arg, Arrays.copyOfRange(args, 1, 4));

        arg = ArgUtils.getStringFromArg(args, 4);
        assertEquals(2, arg.length);
        assertArrayEquals(arg, Arrays.copyOfRange(args, 5, 7));
    }

    @Test
    public void testNumSubArgs() {
        int numSubArgs = ArgUtils.numSubArgs(args, 0);
        assertEquals(3, numSubArgs);

        numSubArgs = ArgUtils.numSubArgs(args, 4);
        assertEquals(2, numSubArgs);
    }

}
