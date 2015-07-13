package com.kushalarora.compositionalLM.utils;

import lombok.val;

/**
 * Created by karora on 6/17/15.
 */
public class ArgUtils {
    /**
     * Returns the array of filenames for the current argument.
     *
     * @param args     array of arguments.
     * @param argIndex index of the current argument being processed.
     * @return An array of filenames.
     */
    public static String[] getStringFromArg(String[] args, int argIndex) {
        val numSubArgs = ArgUtils.numSubArgs(args, argIndex);
        String[] strings = new String[numSubArgs];

        for (int i = 0; i < numSubArgs; i++) {
            strings[i] = args[argIndex + 1 + i].trim();
        }
        return strings;
    }

    /**
     * Helper function to count the number of sub arguments
     *
     * @param args     array of arguments
     * @param argIndex index of current argument
     * @return count of sub arguments for current argument.
     */
    public static int numSubArgs(String[] args, int argIndex) {
        int index = argIndex;
        while (index + 1 < args.length && args[index + 1].charAt(0) != '-') {
            index++;
        }
        return index - argIndex;
    }
}
