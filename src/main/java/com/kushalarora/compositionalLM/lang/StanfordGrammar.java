package com.kushalarora.compositionalLM.lang;

import edu.stanford.nlp.ling.HasContext;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.parser.common.ParserConstraint;
import edu.stanford.nlp.parser.lexparser.*;
import edu.stanford.nlp.parser.lexparser.Options;
import edu.stanford.nlp.util.Index;
import lombok.extern.slf4j.Slf4j;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.regex.Matcher;

import static edu.berkeley.nlp.math.SloppyMath.logAdd;
import static java.lang.Math.*;

/**
 * Most of this code is copied from Stanford ExhaustiveParser.java
 * and modified to compute inside span score and outside span score.
 * Created by karora on 6/21/15. I have removed all the extra code
 * handling dependency parsing and tags handling to keep it simple
 * and clean.
 *
 * @author Kushal Arora
 */
@Slf4j
public class StanfordGrammar extends ExhaustivePCFGParser implements IGrammar {

    protected float[][][] iSpanSplitScore;
    protected float[][] iSpanScore;
    protected float[][] oSpanScore;

    com.kushalarora.compositionalLM.options.Options op;

    public StanfordGrammar(BinaryGrammar bg, UnaryGrammar ug, Lexicon lex,
                           com.kushalarora.compositionalLM.options.Options op,
                           Options defaultOp,   // TODO:: Deprecate this
                           Index<String> stateIndex, Index<String> wordIndex,
                           Index<String> tagIndex) {
        super(bg, ug, lex, defaultOp, stateIndex, wordIndex, tagIndex);
        this.op = op;
    }


    public float[][][] getInsideSpanSplitProb() {
        return iSpanSplitScore;
    }

    public float[][] getInsideSpanProb() {
        return iSpanScore;
    }


    public float[][] getOutsideSpanProb() {
        return oSpanScore;
    }

    public List<Word> getCurrentSentence() {
        return sentence;
    }

    /**
     * Compute inside and outside score for the sentence.
     * Also computes span and span split score we need.
     * @param sentence Sentence being processed.
     */
    public void computeInsideOutsideProb(List<Word> sentence) {
        this.sentence = sentence;
        length = sentence.size();

        log.info("Computing Inside Outside Score for {}", sentence);
        if (length > arraySize) {
            considerCreatingArrays(length);
        }

        log.debug("Initializing arrays with 0f");
        for (int start = 0; start < length; start++) {
            for (int end = start + 1; end <= length; end++) {
                Arrays.fill(iScore[start][end], 0f);
                Arrays.fill(oScore[start][end], 0f);
                Arrays.fill(iSpanSplitScore[start][end], 0f);

                oSpanScore[start][end] = 0f;
            }
        }

        /*log.debug("Initializing narrowLExtent and wideLExtent arrays with {} and {}", -1, length +1);
        for (int loc = 0; loc <= length; loc++) {
            // the rightmost left with state s ending at i that we can get is the beginning
            Arrays.fill(narrowLExtent[loc], -1);
            // the leftmost left with state s ending at i that we can get is the end
            Arrays.fill(wideLExtent[loc], length + 1);
        }
        log.debug("Initializing narrowRExtent and wideRExtent arrays with {} and {}", length +1, -1);
        for (int loc = 0; loc < length; loc++) {
            // the leftmost right with state s starting at i that we can get is the end
            Arrays.fill(narrowRExtent[loc], length + 1);
            // the rightmost right with state s starting at i that we can get is the beginning
            Arrays.fill(wideRExtent[loc], -1);
        }*/
        initializeChart(sentence);

        log.debug("Starting inside score computation");
        doInsideScores();


        log.debug("Start outside score computation");
        int goal = stateIndex.indexOf(goalStr);
        oScore[0][length][goal] = 1.0f;
        doOutsideScores();
    }


    /**
     * Conditionally create inside, outside, [narrow|wide][R|L]Extent arrays,
     * iSpanSplitScore, oSpanScore. If unable to create array try restore the current size.
     *
     * @param length Length of the sentence.
     */

    // Kushal::Method private in base class
    protected void considerCreatingArrays(int length) {
        if (length > op.grammarOp.maxLength + 1 || length >= myMaxLength) {
            throw new OutOfMemoryError("Refusal to create such large arrays.");
        } else {
            try {
                createArrays(length + 1);
            } catch (OutOfMemoryError e) {
                myMaxLength = length;
                if (arraySize > 0) {
                    try {
                        createArrays(arraySize);
                    } catch (OutOfMemoryError e2) {
                        throw new RuntimeException("CANNOT EVEN CREATE ARRAYS OF ORIGINAL SIZE!!");
                    }
                }
                throw e;
            }
            arraySize = length + 1;
            log.debug("Created PCFG parser arrays of size " + arraySize);
        }
    }

    /**
     * Create inside, outside, [narrow|wide][R|L]Extent arrays,
     * iSpanSplitScore, oSpanScore.
     *
     * @param length Length of the sentence
     */
    // Kushal::Added initialization code for span scores
    protected void createArrays(int length) {
        // zero out some stuff first in case we recently
        // ran out of memory and are reallocating
        clearArrays();

        // allocate just the parts of iScore and oScore used (end > start, etc.)
        // todo: with some modifications to doInsideScores,
        // we wouldn't need to allocate iScore[i,length] for i != 0 and i != length
        log.debug("initializing iScore arrays with length " + length + " and numStates " + numStates);
        iScore = new float[length][length + 1][];
        for (int start = 0; start < length; start++) {
            for (int end = start + 1; end <= length; end++) {
                iScore[start][end] = new float[numStates];
            }
        }

        log.debug("initializing oScore arrays with length " + length + " and numStates " + numStates);
        oScore = new float[length][length + 1][];
        for (int start = 0; start < length; start++) {
            for (int end = start + 1; end <= length; end++) {
                oScore[start][end] = new float[numStates];
            }
        }

        // Kushal::ADDED CODE BEGIN
        log.debug("initializing iSpanSplitScore arrays with length " + length);
        iSpanSplitScore = new float[length][length + 1][];
        for (int start = 0; start < length; start++) {
            for (int end = start + 1; end <= length; end++) {
                iSpanSplitScore[start][end] = new float[length];
            }
        }

        log.debug("initializing oSpanScore arrays with length " + length);
        oSpanScore = new float[length][length + 1];
        // Kushal::ADDED CODE END
/*

        narrowRExtent = new int[length][numStates];
        wideRExtent = new int[length][numStates];
        narrowLExtent = new int[length + 1][numStates];
        wideLExtent = new int[length + 1][numStates];

*/
        log.debug("Finished creating iscore, oscore, narrow, wide, " +
                "iSpanSplit and iSpan arrays");
    }

    /**
     * Deallocate all the arrays.
     */
    // Kushal::Added nullification of span score arrays
    protected void clearArrays() {
        log.debug("clearing arrays");
        iScore = oScore = null;
        /*narrowRExtent = wideRExtent = narrowLExtent = wideLExtent = null;*/
        // Kushal::ADDED CODE BEGIN
        iSpanSplitScore = null;
        oSpanScore = null;
        // Kushal::ADDED CODE END
    }

    /**
     * Add unary rules to the chart.
     * Adds non terminal to terminal unary rules first and
     * then handles the non terminal to non terminal rules by
     * marginalizing by non-terminal children to given parent i.e.
     * by summing all iScore of non terminal children symbols and
     * adding it to parent score.
     * <p/>
     * iSpanSplitScore is nothing but the sum over all states and
     * iSpanScore is nothing but the same as span is of size 1 and
     * there is only one possible split
     *
     * @param sentence sentence to use to initializeChart
     */
    // Kushal::Method private in base class
    // Kushal::Added code to fill span arrays for words using unary rules
    protected void initializeChart(List<? extends HasWord> sentence) {
        for (int start = 0; start < length; start++) {
            int word = words[start];
            int end = start + 1;
            Arrays.fill(tags[start], false);

            float[] iScore_start_end = iScore[start][end];
            /*int[] narrowRExtent_start = narrowRExtent[start];
            int[] narrowLExtent_end = narrowLExtent[end];
            int[] wideRExtent_start = wideRExtent[start];
            int[] wideLExtent_end = wideLExtent[end];*/

            //Word context (e.g., morphosyntactic info)
            // TODO:: Figure out how to use this to advantage
            String wordContextStr = null;
            if (sentence.get(start) instanceof HasContext) {
                wordContextStr = ((HasContext) sentence.get(start)).originalText();
                if ("".equals(wordContextStr))
                    wordContextStr = null;
            }

            boolean assignedSomeTag = false;
            // For each word, figure out the corresponding tag,
            // We are assuming here that lexicon has a list of tags
            // corresponding to each word
            // Use lexicon to get p(word|tag).
            // This is basically N => word unary rule.
            for (Iterator<IntTaggedWord> taggingI = lex.ruleIteratorByWord(word, start, wordContextStr); taggingI.hasNext(); ) {
                IntTaggedWord tagging = taggingI.next();
                int state = stateIndex.indexOf(tagIndex.get(tagging.tag));
                // score the cell according to P(word|tag) in the lexicon
                float lexScore = lex.score(tagging, start, wordIndex.get(tagging.word), wordContextStr);
                if (lexScore > Float.NEGATIVE_INFINITY) {
                    assignedSomeTag = true;
                    iScore_start_end[state] = ((float)exp(lexScore));
                    /*
                    // leftmost and rightmost right child of
                    // non terminal state with span starting at
                    // start are equal and span one word *word*
                    narrowRExtent_start[state] = end;
                    wideRExtent_start[state] = end;

                    // rightmost and leftmost left child of
                    // non terminal state with span ending at
                    // end(start+1) are equal and span one word *word*
                    narrowLExtent_end[state] = start;
                    wideLExtent_end[state] = start;
                    */
                }

                int tag = tagging.tag;
                tags[start][tag] = true;
            }

            if (!assignedSomeTag) {
                // In case the list of tags for words is missing,
                // we give words all tags for which
                // the lexicon score is not -Inf,
                // not just seen or specified taggings

                for (int state = 0; state < numStates; state++) {
                    if (isTag[state] && iScore_start_end[state] == Float.NEGATIVE_INFINITY) {

                        float lexScore = lex.score(new IntTaggedWord(word, tagIndex.indexOf(stateIndex.get(state))), start, wordIndex.get(word), wordContextStr);
                        if (lexScore > Float.NEGATIVE_INFINITY) {
                            iScore_start_end[state] = ((float)exp(lexScore));
                            /*
                            // leftmost and rightmost right child of
                            // non terminal state with span starting at
                            // start are equal and span one word *word*
                            narrowRExtent_start[state] = end;
                            wideRExtent_start[state] = end;

                            // rightmost and leftmost left child of
                            // non terminal state with span ending at
                            // end(start+1) are equal and span one word *word*
                            narrowLExtent_end[state] = start;
                            wideLExtent_end[state] = start;
                            */
                        }
                    }
                }
            } // end if ! assignedSomeTag

            // Apply unary rules in diagonal cells of chart
            for (int state = 0; state < numStates; state++) {

                double iS = iScore_start_end[state];
                // Unary rules are from non-terminal to non-terminal
                // if no non-terminal spans (start, end), nothing to
                // expand here
                if (iS == 0) {
                    continue;
                }
                iS = log(iS);


                UnaryRule[] unaries = ug.closedRulesByChild(state);
                for (UnaryRule ur : unaries) {
                    int parentState = ur.parent;
                    float pS = ur.score;
                    double tot = exp(logAdd(iS, pS));
                    // A parentNode might connect to the span via
                    // different intermediate tags, so adding to
                    // previous instead of overwriting
                    iScore_start_end[parentState] += ((float) tot);

                    /*
                    // Min and max left and right span for
                    // parentState can be start(end) and end(start)
                    // as only span one has been considered till now
                    // so not overwriting anything
                    narrowRExtent_start[parentState] = end;
                    narrowLExtent_end[parentState] = start;

                    wideRExtent_start[parentState] = end;
                    wideLExtent_end[parentState] = start;*/

                }   // end for unary rules
            }   // end for state for unary rules


            for (int state = 0; state < numStates; state++) {
                // iSpanSplitScore[start][end][split] is nothing but marginalization
                // over all states
                iSpanSplitScore[start][end][start] += iScore_start_end[state];
            }

            // iSpanSplitScore[start][end][start] is nothing but marginalization
            // over all splits
            iSpanScore[start][end] = iSpanSplitScore[start][end][start];

        } // end for start
    } // end initializeChart(List sentence)

    /**
     * Fills in the iScore array of each category over each span
     * of length 2 or more.
     */
    protected void doInsideScores() {
        // TODO::Question: Do we really need to go till length or
        // stop before.
        for (int start = 0; start <= length - 2; start++) {
            for (int end = start + 2; end <= length; end++) {
                doInsideChartCell(start, end);
            }
        }
    } // end doInsideScores()


    /**
     * Compute inside, inside span, inside span split Score for span (start,end).
     * @param start start index of span
     * @param end end index of span
     */
    private void doInsideChartCell(final int start, final int end) {
        /* TODO: Figure out later if using these constraints can help us anyway. */
        final List<ParserConstraint> constraints = getConstraints();
        if (constraints != null) {
            for (ParserConstraint c : constraints) {
                if ((start > c.start && start < c.end && end > c.end) || (end > c.start && end < c.end && start < c.start)) {
                    return;
                }
            }
        }

        /*int[] narrowRExtent_start = narrowRExtent[start];
        int[] wideRExtent_start = wideRExtent[start];
        int[] narrowLExtent_end = narrowLExtent[end];
        int[] wideLExtent_end = wideLExtent[end];*/
        float[][] iScore_start = iScore[start];
        float[] iScore_start_end = iScore_start[end];

        for (int parentState = 0; parentState < numStates; parentState++) {
            List<BinaryRule> parentRules = bg.ruleListByParent(parentState);
            for (BinaryRule rule : parentRules) {

                int leftChild = rule.leftChild;
                int rightChild = rule.rightChild;

                /*// leftmost right sibling of the left child
                int narrowROfLC = narrowRExtent_start[leftChild];
                // rightmost right sibling of left child
                int wideROfLC = wideRExtent_start[leftChild];

                // rightmost left sibling of the right child
                int narrowLOfRC = narrowLExtent_end[rightChild];
                // leftmost left sibling of right child
                int wideLOfRC = wideLExtent_end[rightChild];

                // can this left constituent leave space for a right constituent?
                if (narrowROfLC >= end) {
                    // maybe throw RunTimeException
                    continue;
                }

                // can this right constituent leave space for a left constituent?
                if (narrowLOfRC <= start) {
                    // maybe throw RunTimeException
                    continue;
                }

                // can this right constituent fit next to the left constituent?
                // rightmost left child corresponding to right sibling is positioned before
                // leftmost right child corresponding to the left sibling
                // The two children can't fit next to each other as they always
                // overlap
                if (narrowLOfRC < narrowROfLC) {
                    // maybe throw RunTimeException
                    continue;
                }

                // maxPossibleLeftSpan = max(leftmost right sibling of left child,
                //                           leftmost left sibling of the right child)
                //  interior of the two
                int maxPossibleLeftSpan = max(narrowROfLC, wideLOfRC);

                // maxPossibleRightSpan = min(rightmost right sibling of leftchild,
                //                            rightmost left sibling of the right child)
                //  interior of the two
                int maxPossibleRightSpan = (wideROfLC < narrowLOfRC ? wideROfLC : narrowLOfRC);

                // can this left constituent stretch far enough to reach the right constituent?
                // If maximum possible right span left child is to the left of
                // max possible left right child, then there is no chance
                // these rules can cover the whole span
                if (maxPossibleLeftSpan > maxPossibleRightSpan) {
                    continue;
                }*/

                // This binary split might be able to cover the span depending upon children's coverage.
                float pS = rule.score;

                //System.out.println("Min "+maxPossibleLeftSpan+" max "+maxPossibleRightSpan+" start "+start+" end "+end);

                // find the split that can use this rule to make the max score
                for (int split = start; split < end; split++) {

                    if (constraints != null) {
                        boolean skip = false;
                        for (ParserConstraint c : constraints) {
                            if (((start < c.start && end >= c.end) || (start <= c.start && end > c.end)) && split > c.start && split < c.end) {
                                skip = true;
                                break;
                            }
                            if ((start == c.start && split == c.end)) {
                                String tag = stateIndex.get(leftChild);
                                Matcher m = c.state.matcher(tag);
                                if (!m.matches()) {
                                    skip = true;
                                    break;
                                }
                            }
                            if ((split == c.start && end == c.end)) {
                                String tag = stateIndex.get(rightChild);
                                Matcher m = c.state.matcher(tag);
                                if (!m.matches()) {
                                    skip = true;
                                    break;
                                }
                            }
                        }
                        if (skip) {
                            continue;
                        }
                    }

                    double lS = iScore_start[split][leftChild];
                    if (lS == 0f) {
                        continue;
                    }
                    lS = log(lS);

                    double rS = iScore[split][end][rightChild];
                    if (rS == 0f) {
                        continue;
                    }
                    rS = log(rS);

                    /*
                    The stanford Parser searches for best
                    parse. Our objective is different and is
                    to compute the Inside Score like in
                    Larry and Young's papar.
                    */
                    double tot = exp(logAdd(new double[]{pS, lS, rS}));
                    iScore_start_end[parentState] += tot;

                } // for split point
            } // end for leftRules
        }

        // do unary rules -- one could promote this loop and put start inside
        for (int state = 0; state < numStates; state++) {

            double iS = iScore_start_end[state];
            if (iS == 0f) {
                continue;
            }
            iS = log(iS);

            UnaryRule[] unaries = ug.closedRulesByChild(state);
            for (UnaryRule ur : unaries) {

                if (constraints != null) {
                    boolean skip = false;
                    for (ParserConstraint c : constraints) {
                        if ((start == c.start && end == c.end)) {
                            String tag = stateIndex.get(ur.parent);
                            Matcher m = c.state.matcher(tag);
                            if (!m.matches()) {
                                //if (!tag.startsWith(c.state+"^")) {
                                skip = true;
                                break;
                            }
                        }
                    }
                    if (skip) {
                        continue;
                    }
                }

                int parentState = ur.parent;
                float pS = ur.score;
                double tot = exp(logAdd(iS, pS));
                iScore_start_end[parentState] = ((float) tot);

                /*double cur = iScore_start_end[parentState];
                if (cur == 0f) {
                    if (start > narrowLExtent_end[parentState]) {
                        narrowLExtent_end[parentState] = wideLExtent_end[parentState] = start;
                    } else if (start < wideLExtent_end[parentState]) {
                        wideLExtent_end[parentState] = start;
                    }
                    if (end < narrowRExtent_start[parentState]) {
                        narrowRExtent_start[parentState] = wideRExtent_start[parentState] = end;
                    } else if (end > wideRExtent_start[parentState]) {
                        wideRExtent_start[parentState] = end;
                    }
                }*/
            } // for UnaryRule r
        } // for unary rules
    }


    private void doOutsideScores() {
        for (int diff = length; diff >= 1; diff--) {
            for (int start = 0; start + diff <= length; start++) {
                int end = start + diff;

                // do unaries
                for (int state = 0; state < numStates; state++) {
                    double oS = oScore[start][end][state];
                    if (oS == 0f) {
                        continue;
                    }
                    oS = log(oS);

                    UnaryRule[] rules = ug.closedRulesByParent(state);
                    for (UnaryRule ur : rules) {
                        float pS = ur.score;
                        double tot = exp(logAdd(oS, pS));
                        oScore[start][end][ur.child] += ((float)tot);
                    }
                }

                // do binaries
                for (int state = 0; state < numStates; state++) {
                    /*int min1 = narrowRExtent[start][state];
                    if (end < min1) {
                        continue;
                    }*/
                    BinaryRule[] rules = bg.splitRulesWithLC(state);
                    for (BinaryRule br  : rules) {

                        double oS = oScore[start][end][br.parent];
                        if (oS == 0f) {
                            continue;
                        }
                        oS = log(oS);

                        /*int max1 = narrowLExtent[end][br.rightChild];
                        if (max1 < min1) {
                            continue;
                        }

                        int min = min1;
                        int max = max1;

                        if (max - min > 2) {
                            int min2 = wideLExtent[end][br.rightChild];
                            min = (min1 > min2 ? min1 : min2);
                            if (max1 < min) {
                                continue;
                            }

                            int max2 = wideRExtent[start][br.leftChild];
                            max = (max1 < max2 ? max1 : max2);
                            if (max < min) {
                                continue;
                            }
                        }*/
                        float pS = br.score;

                        for (int split = start; split < end; split++) {

                            double rS = iScore[split][end][br.rightChild];
                            if (rS == 0f) {
                                continue;
                            }
                            double totR = exp(logAdd(new double[] {pS, rS, oS}));
                            oScore[split][end][br.rightChild] += ((float)totR);
                        }
                    }
                }
                for (int state = 0; state < numStates; state++) {
                    /*int max1 = narrowLExtent[end][state];
                    if (max1 < start) {
                        continue;
                    }*/
                    BinaryRule[] rules = bg.splitRulesWithRC(state);
                    for (BinaryRule br : rules) {
                        float oS = oScore[start][end][br.parent];
                        if (oS == 0f) {
                            continue;
                        }

                        /*int min1 = narrowRExtent[start][br.leftChild];
                        if (max1 < min1) {
                            continue;
                        }

                        int min = min1;
                        int max = max1;
                        if (max - min > 2) {
                            int min2 = wideLExtent[end][br.rightChild];
                            min = (min1 > min2 ? min1 : min2);
                            if (max1 < min) {
                                continue;
                            }
                            int max2 = wideRExtent[start][br.leftChild];
                            max = (max1 < max2 ? max1 : max2);
                            if (max < min) {
                                continue;
                            }
                        }*/

                        float pS = br.score;

                        for (int split = start; split <= end; split++) {
                            double lS = iScore[start][split][br.leftChild];
                            if (lS == 0f) {
                                continue;
                            }
                            lS = log(lS);

                            double totL = exp(logAdd(new double[] {pS, lS, oS}));
                            oScore[start][split][br.leftChild] += ((float)totL);
                        }
                    }
                }
            }
        }
    }

}

