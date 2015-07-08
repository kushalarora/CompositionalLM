package com.kushalarora.compositionalLM.lang;

import com.kushalarora.compositionalLM.options.GrammarOptions;
import edu.stanford.nlp.ling.HasContext;
import edu.stanford.nlp.parser.lexparser.*;
import edu.stanford.nlp.util.Index;
import lombok.extern.slf4j.Slf4j;
import lombok.val;

import java.util.*;

import static java.lang.Math.exp;
import static java.lang.Math.log;

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
    /**
     * Created by karora on 6/24/15.
     */
    public class StanfordInsideOutsideScore extends AbstractInsideOutsideScores {

        private transient float[][][][] iSplitSpanStateScore;
        private transient float[][][][] oSpanStateScoreWParent;

        public StanfordInsideOutsideScore(List sentence) {
            super(sentence);
        }


        /**
         * Deallocate all the arrays.
         */
        // Kushal::Added nullification of span score arrays
        public void clearArrays() {
            log.debug("clearing arrays");
            // [start][end][state]
            iScore = null;

            //[start][end]
            iSpanScore = null;

            // [start][end][split]
            iSpanSplitScore = null;

            // [start][end][split][state]
            iSplitSpanStateScore = null;

            // [start][end][state]
            oScore = null;

            // [start][end]
            oSpanScore = null;

            // [start][end][parent]
            oSpanWParentScore = null;

            // [start][end][parent][state]
            oSpanStateScoreWParent = null;

            // [start][end][state]
            muScore = null;

            // [start][end][split]
            muSpanSplitScore = null;

            // [start][end][split][parent]
            muSpanScoreWParent = null;
        }

        /**
         * Conditionally create inside, outside, [narrow|wide][R|L]Extent arrays,
         * iSpanSplitScore, oSpanScore. If unable to create array try restore the current size.
         *
         * @param length Length of the sentence.
         */

        // Kushal::Method private in base class
        private void considerCreatingArrays(int length) {
            // maxLength + 1 as we added boundary symbol to sentence
            if (length > GrammarOptions.maxLength + 1
                    // myMaxLength if greater than zero,
                    // then it is max memory size
                    || length >= myMaxLength) {
                throw new OutOfMemoryError("Refusal to create such large arrays.");
            } else {
                try {
                    // Kushal::TODO: Stanford created array with length + 1
                    // as they had do deal with boundary smybol.
                    // Figure out if we need to deal too.
                    // Till then creating array of length instead
                    createArrays(length);
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
                arraySize = length;
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
        private void createArrays(int length) {
            // zero out some stuff first in case we recently
            // ran out of memory and are reallocating
            clearArrays();

            // allocate just the parts of iScore and oScore used (end > start, etc.)
            // todo: with some modifications to doInsideScores,
            // we wouldn't need to allocate iScore[i,length] for i != 0 and i != length
            iScore = new float[length][length + 1][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    iScore[start][end] = new float[numStates];
                }
            }

            oScore = new float[length][length + 1][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    oScore[start][end] = new float[numStates];
                }
            }

            iSpanScore = new float[length][length + 1];
            oSpanScore = new float[length][length + 1];

            iSpanSplitScore = new float[length][length + 1][];
            oSpanWParentScore = new float[length][length + 1][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    // splits
                    iSpanSplitScore[start][end] = new float[length];
                    // parents
                    oSpanWParentScore[start][end] = new float[length + 1];
                }
            }

            oSpanStateScoreWParent = new float[length][length + 1][][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    // parents
                    oSpanStateScoreWParent[start][end] = new float[length + 1][];
                    for (int parent = 0; parent < start; parent++) {
                        // states
                        oSpanStateScoreWParent[start][end][parent] = new float[numStates];
                    }

                    for (int parent = end; parent <= length; parent++) {
                        // states
                        oSpanStateScoreWParent[start][end][parent] = new float[numStates];
                    }
                }
            }

            iSplitSpanStateScore = new float[length][length + 1][][];
            muSpanScoreWParent = new float[length][length + 1][][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    // splits
                    muSpanScoreWParent[start][end] = new float[length][];
                    iSplitSpanStateScore[start][end] = new float[length][];
                    for (int split = start; split < end; split++) {
                        // states
                        iSplitSpanStateScore[start][end][split] = new float[numStates];
                        // parents
                        muSpanScoreWParent[start][end][split] = new float[length + 1];
                    }
                }
            }

            muScore = new float[length][length + 1][];
            muSpanSplitScore = new float[length][length + 1][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    // splits
                    muSpanSplitScore[start][end] = new float[length];
                    // states
                    muScore[start][end] = new float[numStates];
                }
            }
            log.debug("Finished allocating inside, outside and mu score arrays");
        }

        @Override
        public void initializeScoreArrays() {
            log.info("Computing Inside Outside Score for {}", sentence);
            if (length > arraySize) {
                considerCreatingArrays(length);
            }


            log.debug("Initializing arrays with 0f");
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    // fill states
                    Arrays.fill(iScore[start][end], 0f);
                    Arrays.fill(oScore[start][end], 0f);

                    iSpanScore[start][end] = 0f;
                    oSpanScore[start][end] = 0f;

                    // fill splits (start, end)
                    Arrays.fill(iSpanSplitScore[start][end], 0f);
                    // fill splits (start, end)
                    Arrays.fill(oSpanWParentScore[start][end], 0f);

                    for (int split = start + 1; split < end; split++) {
                        // fill state
                        Arrays.fill(iSplitSpanStateScore[start][end][split], 0f);

                        // fill parent
                        Arrays.fill(muSpanScoreWParent[start][end][split], 0f);
                    }

                    for (int parent = 0; parent < start; parent++) {
                        // fill state
                        Arrays.fill(oSpanStateScoreWParent[start][end][parent], 0f);
                    }

                    for (int parent = end; parent <= length; parent++) {
                        // fill state
                        Arrays.fill(oSpanStateScoreWParent[start][end][parent], 0f);
                    }

                    // fill state
                    Arrays.fill(muScore[start][end], 0f);

                    // fill split
                    Arrays.fill(muSpanSplitScore[start][end], 0f);
                }
            }
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
         */
        // Kushal::Method private in base class
        // Kushal::Removed all node for narrow, wide caching
        // Kushal::Added code to fill span arrays for words using unary rules
        // TODO:: Currently span and split span are unnormalized as we get p(word|tag)
        public void doLexScores() {
            words = new int[length];
            tags = new boolean[length][numStates];

            for (int i = 0; i < length; i++) {
                String s = sentence.get(i).word();
                words[i] = wordIndex.addToIndex(s);
            }

            for (int start = 0; start < length; start++) {
                int word = words[start];
                int end = start + 1;
                Arrays.fill(tags[start], false);

                float[] iScore_start_end = iScore[start][end];

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
                Iterator<IntTaggedWord> taggingI;
                for (taggingI = lex.ruleIteratorByWord(word, start, wordContextStr);
                     taggingI.hasNext(); ) {

                    IntTaggedWord tagging = taggingI.next();
                    int state = stateIndex.indexOf(tagIndex.get(tagging.tag));
                    // score the cell according to P(word|tag) in the lexicon
                    float lexScore = lex.score(tagging, start,
                            wordIndex.get(tagging.word), wordContextStr);

                    if (lexScore > Float.NEGATIVE_INFINITY) {
                        assignedSomeTag = true;
                        double tot = exp(lexScore);
                        iScore_start_end[state] += tot;

                        // in case of leaf nodes there is no node, hence
                        // we keeping the value of split at start
                        iSpanSplitScore[start][end][start] += tot;
                        iSplitSpanStateScore[start][end][start][state] += tot;

                        iSpanScore[start][end] += tot;
                        log.debug("Start:{} End:{} State: {}  Tag => Word : {} {} Score: {}", start, end, state,
                                tagging,
                                lexScore,
                                iScore_start_end[state]);
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

                            float lexScore = lex.score(new IntTaggedWord(word,
                                            tagIndex.indexOf(stateIndex.get(state))),
                                    start, wordIndex.get(word), wordContextStr);
                            if (lexScore > Float.NEGATIVE_INFINITY) {
                                double tot = exp(lexScore);
                                iScore_start_end[state] += tot;

                                // in case of leaf nodes there is no node, hence
                                // we keeping the value of split at start
                                iSpanSplitScore[start][end][start] += tot;
                                iSplitSpanStateScore[start][end][start][state] += tot;

                                iSpanScore[start][end] += tot;
                            }
                        }
                    }
                } // end if ! assignedSomeTag

                // Apply unary rules
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
                        double tot = exp(iS + pS);
                        // A parentNode might connect to the span via
                        // different intermediate tags, so adding to
                        // previous instead of overwriting

                        iScore_start_end[parentState] += ((float) tot);

                        // in case of leaf nodes there is no node, hence
                        // we keeping the value of split at start
                        iSpanSplitScore[start][end][start] += tot;
                        iSplitSpanStateScore[start][end][start][state] += tot;

                        iSpanScore[start][end] += tot;
                        log.debug("Start:{} End:{} Unary {} iScore: {}", start, end, ur,
                                iScore_start_end[parentState]);

                    }   // end for unary rules
                }   // end for state for unary rules

                log.debug("iSpanScore[{}][{}]: {}", start, end, iSpanScore[start][end]);
            } // end for start
        } // end doLexScores(List sentence)

        /**
         * Fills in the iScore array of each category over each span
         * of length 2 or more.
         */
        public void doInsideScores() {
            for (int diff = 2; diff <= length; diff++) {
                // usually stop one short because boundary symbol only combines
                // with whole sentence span. So for 3 word sentence + boundary = 4,
                // length == 4, and do [0,2], [1,3]; [0,3]; [0,4]
                for (int start = 0; start < ((diff == length) ? 1 : length - diff); start++) {
                    doInsideChartCell(start, start + diff);
                } // for start
            } // for diff (i.e., span)
        } // end doInsideScores()


        /**
         * Compute inside, inside span, inside span split Score for span (start,end).
         *
         * @param start start index of span
         * @param end   end index of span
         */
        private void doInsideChartCell(final int start, final int end) {
            boolean[][] stateSplit = new boolean[numStates][length];
            Set<BinaryRule> binaryRuleSet = new HashSet<BinaryRule>();
            for (int leftState = 0; leftState < numStates; leftState++) {
                BinaryRule[] leftRules = bg.splitRulesWithLC(leftState);
                for (BinaryRule rule : leftRules) {

                    int rightChild = rule.rightChild;
                    int parentState = rule.parent;

                    // This binary split might be able to cover the span depending upon children's coverage.
                    float pS = rule.score;


                    // calculate iScore for state by summing over split
                    // and iSpanSplitScore by summing over states
                    // Stops one short of end as last span should be
                    // (start, end - 1), (end - 1, end)
                    for (int split = start + 1; split < end; split++) {

                        double lS = iScore[start][split][leftState];
                        if (lS == 0f) {
                            continue;
                        }
                        lS = log(lS);

                        double rS = iScore[split][end][rightChild];
                        if (rS == 0f) {
                            continue;
                        }
                        rS = log(rS);

                        binaryRuleSet.add(rule);

                        stateSplit[parentState][split] = true;

                        double tot = exp(pS + lS + rS);
                        // in left child
                        iScore[start][end][parentState] += tot;

                        // Marginalizing over parentState and retaining
                        // split index
                        iSpanSplitScore[start][end][split] += tot;

                        iSplitSpanStateScore[start][end][split][parentState] += tot;
                    } // for split point
                } // end for leftRules
            }

            for (int rightState = 0; rightState < numStates; rightState++) {
                BinaryRule[] rightRules = bg.splitRulesWithRC(rightState);
                for (BinaryRule rule : rightRules) {

                    // Rule already processed by leftstate loop
                    if (binaryRuleSet.contains(rule)) {
                        log.debug("Rule {} already processed by left child loop.Skipping", rule);
                        continue;
                    }

                    int leftChild = rule.leftChild;
                    int parentState = rule.parent;

                    // This binary split might be able to cover the span depending upon children's coverage.
                    float pS = rule.score;


                    // calculate iScore for state by summing over split
                    // and iSpanSplitScore by summing over states
                    // Stops one short of end as last span should be
                    // (start, end - 1), (end - 1, end)
                    for (int split = start + 1; split < end; split++) {

                        double lS = iScore[start][split][leftChild];
                        if (lS == 0f) {
                            continue;
                        }
                        lS = log(lS);

                        double rS = iScore[split][end][rightState];
                        if (rS == 0f) {
                            continue;
                        }
                        rS = log(rS);

                        binaryRuleSet.add(rule);

                        stateSplit[parentState][split] = true;

                        double tot = exp(pS + lS + rS);
                        // right child
                        iScore[start][end][parentState] += tot;


                        // Marginalizing over parentState and retaining
                        // split index
                        iSpanSplitScore[start][end][split] += tot;

                        iSplitSpanStateScore[start][end][split][parentState] += tot;
                    } // for split point
                } // end for rightRules
            }

            // do unary rules -- one could promote this loop and put start inside
            for (int state = 0; state < numStates; state++) {

                double iS = iScore[start][end][state];
                if (iS == 0f) {
                    continue;
                }
                iS = log(iS);

                UnaryRule[] unaries = ug.closedRulesByChild(state);
                for (UnaryRule ur : unaries) {

                    int parentState = ur.parent;
                    float pS = ur.score;
                    double tot = exp(iS + pS);
                    iScore[start][end][parentState] += tot;

                    // Kushal::ADDED CODE BEGINS
                    // We are marginalizing over all states and this state
                    // spans (start,end) and should be marginalized for all
                    // splits
                    for (int split = start + 1; split < end; split++) {
                        if (stateSplit[state][split]) {
                            iSpanSplitScore[start][end][split] += tot;
                            iSplitSpanStateScore[start][end][split][parentState] += tot;
                        }
                    }
                    //  Kushal::ADDED CODE ENDS

                } // for UnaryRule r
            } // for unary rules

            // Kushal::ADDED CODE BEGINS
            for (int split = start + 1; split < end; split++) {
                // Marginalizing over both split and parent state
                iSpanScore[start][end] += iSpanSplitScore[start][end][split];
            }
            // Kushal::ADDED CODE ENDS

            for (int state = 0; state < numStates; state++) {
                if (iScore[start][end][state] > 0) {
                    log.debug("iScore[{}][{}][{}] = {}", start, end, state, iScore[start][end][state]);
                }
            }
        }

        /**
         *
         */
        public void doOutsideScores() {
            int goal = stateIndex.indexOf(goalStr);
            oScore[0][length][goal] = 1.0f;
            oSpanScore[0][length] = 1.0f;
            oSpanWParentScore[0][length][length] = 1.0f;
            oSpanStateScoreWParent[0][length][length][goal] = 1.0f;

            for (int diff = length; diff >= 1; diff--) {
                for (int start = 0; start + diff <= length; start++) {
                    int end = start + diff;

                    log.info("Computing oScore for span ({}, {})", start, end);
                    // do unaries
                    for (int parentState = 0; parentState < numStates; parentState++) {
                        double oS = oScore[start][end][parentState];
                        if (oS == 0f) {
                            continue;
                        }
                        oS = log(oS);

                        UnaryRule[] rules = ug.closedRulesByParent(parentState);
                        for (UnaryRule ur : rules) {
                            float pS = ur.score;
                            double tot = exp(oS + pS);
                            log.debug("Adding unary rule {} to outside score for Start: {}, End: {}"
                                    , ur, start, end);
                            oScore[start][end][ur.child] += tot;
                            oSpanScore[start][end] += tot;

                            // As this is unary rule and parent span is same as child,
                            // so consider the whole sentence to be parent ending with end
                            oSpanWParentScore[start][end][end] += tot;
                            oSpanStateScoreWParent[start][end][end][ur.child] += tot;
                        }   // end for unary rule iter
                    }   // end for parentState

                    // do binaries
                    for (int leftState = 0; leftState < numStates; leftState++) {
                        BinaryRule[] rules = bg.splitRulesWithLC(leftState);
                        for (BinaryRule br : rules) {

                            double oS = oScore[start][end][br.parent];
                            if (oS == 0f) {
                                continue;
                            }
                            oS = log(oS);

                            float pS = br.score;

                            for (int split = start + 1; split < end; split++) {

                                double rS = iScore[split][end][br.rightChild];
                                if (rS == 0f) {
                                    continue;
                                }
                                rS = log(rS);
                                log.debug("oScore[{}][{}][{}]=oScore[{}][{}][{}]({}) + iScore[{}][{}][{}]({}) Rule {}",
                                        start, split, leftState,
                                        start, end, br.parent, oS,
                                        split, end, br.rightChild, rS,
                                        br);

                                double totR = exp(pS + rS + oS);

                                oScore[start][split][leftState] += totR;

                                // outside Score for span (start, split) with
                                // parent ending at end
                                oSpanWParentScore[start][split][end] += totR;

                                oSpanStateScoreWParent[start][split][end][leftState] += totR;

                                // outside score for span (start, split)
                                oSpanScore[start][split] += totR;

                            }   // end for split
                        }   // end for binary rule iter
                    }   // end for leftState

                    for (int rightState = 0; rightState < numStates; rightState++) {
                        BinaryRule[] rules = bg.splitRulesWithRC(rightState);
                        for (BinaryRule br : rules) {
                            double oS = oScore[start][end][br.parent];
                            if (oS == 0f) {
                                continue;
                            }
                            oS = log(oS);

                            float pS = br.score;

                            for (int split = start + 1; split < end; split++) {
                                double lS = iScore[start][split][br.leftChild];
                                if (lS == 0f) {
                                    continue;
                                }
                                lS = log(lS);

                                log.debug("oScore[{}][{}][{}]=oScore[{}][{}][{}]({}) + iScore[{}][{}][{}]({}) Rule {}",
                                        split, end, rightState,
                                        start, end, br.parent, oS,
                                        start, split, br.leftChild, lS,
                                        br);
                                double totL = exp(pS + lS + oS);
                                oScore[split][end][rightState] += totL;

                                // outside score for span (split, end) with parent
                                // starting at start
                                oSpanWParentScore[split][end][start] += totL;

                                oSpanStateScoreWParent[split][end][start][rightState] += totL;

                                // outside score for span (split, end)
                                oSpanScore[split][end] += totL;

                            }   // end for split
                        }   // end for binary rules iter
                    }   // end for right state

                }   // end for start
            }   // end for end
        }   // end doOutsideScores

        /**
         *
         */
        public void computeMuSpanScore() {

            // Handle lead node case.
            // There is no split here and span value
            // is stored at start
            for (int start = 0; start < length; start++) {
                int end = start + 1;
                int split = start;
                for (int state = 0; state < numStates; state++) {

                    double iS = iScore[start][end][state];
                    if (iS == 0) {
                        continue;
                    }
                    iS = log(iS);

                    double oS = oScore[start][end][state];

                    if (oS == 0f) {
                        continue;
                    }
                    oS = log(oS);

                    double tot;
                    tot = exp(iS + oS);
                    muScore[start][end][state] += tot;
                    log.debug("muScore[{}][{}][{}] = {}",
                            start, end, state, tot);

                    double iSplitSpanStateScore = this.iSplitSpanStateScore[start][end][split][state];
                    if (iSplitSpanStateScore == 0) {
                        continue;
                    }
                    iSplitSpanStateScore = log(iSplitSpanStateScore);

                    for (int parent = 0; parent < start; parent++) {
                        double oScoreWParent = oSpanStateScoreWParent[start][end][parent][state];

                        if (oScoreWParent == 0f) {
                            continue;
                        }
                        oScoreWParent = log(oScoreWParent);
                        tot = exp(oScoreWParent + iSplitSpanStateScore);
                        muSpanScoreWParent[start][end][split][parent] += tot;
                        log.debug("muSpanScoreWParent[{}][{}][{}][{}] = {}",
                                start, end, split, parent, tot);
                    }

                    for (int parent = end; parent < length; parent++) {
                        double oScoreWParent = oSpanStateScoreWParent[start][end][parent][state];

                        if (oScoreWParent == 0f) {
                            continue;
                        }
                        oScoreWParent = log(oScoreWParent);
                        tot = exp(oScoreWParent + iSplitSpanStateScore);
                        muSpanScoreWParent[start][end][split][parent] += tot;
                        log.debug("muSpanScoreWParent[{}][{}][{}][{}] = {}",
                                start, end, split, parent, tot);
                    }
                }
            }

            // Handle cases for diff = 2 and above
            for (int diff = length; diff >= 2; diff--) {
                for (int start = 0; start + diff <= length; start++) {
                    int end = start + diff;
                    for (int state = 0; state < numStates; state++) {

                        double oS = oScore[start][end][state];
                        if (oS == 0) {
                            continue;
                        }
                        oS = log(oS);

                        double iS = iScore[start][end][state];
                        if (iS == 0) {
                            continue;
                        }
                        iS = log(iS);
                        double tot;

                        tot = exp(oS + iS);
                        muScore[start][end][state] += tot;
                        log.debug("muScore[{}][{}][{}] = {}",
                                start, end, state, tot);
                        // This handles the leaf nodes with no split
                        for (int split = start + 1; split < end; split++) {
                            // Marginalizing over both split and parent state
                            double iSplitSpanStateScore = this.iSplitSpanStateScore[start][end][split][state];
                            if (iSplitSpanStateScore == 0) {
                                continue;
                            }
                            iSplitSpanStateScore = log(iSplitSpanStateScore);
                            tot = exp(oS + iSplitSpanStateScore);
                            muSpanSplitScore[start][end][split] += tot;
                            log.debug("muSpanSplitScore[{}][{}][{}] = {}",
                                    start, end, split, tot);

                            for (int parent = 0; parent < start; parent++) {
                                double oScoreWParent = oSpanStateScoreWParent[start][end][parent][state];

                                if (oScoreWParent == 0f) {
                                    continue;
                                }
                                oScoreWParent = log(oScoreWParent);

                                tot = exp(oScoreWParent + iSplitSpanStateScore);
                                muSpanScoreWParent[start][end][split][parent] += tot;
                                log.debug("muSpanScoreWParent[{}][{}][{}][{}] = {}",
                                        start, end, split, parent, tot);
                            }

                            for (int parent = end; parent < length; parent++) {
                                double oScoreWParent = oSpanStateScoreWParent[start][end][parent][state];

                                if (oScoreWParent == 0f) {
                                    continue;
                                }
                                oScoreWParent = log(oScoreWParent);
                                tot = exp(oScoreWParent + iSplitSpanStateScore);
                                muSpanScoreWParent[start][end][split][parent] += tot;
                                log.debug("muSpanScoreWParent[{}][{}][{}][{}] = {}",
                                        start, end, split, parent, tot);
                            }
                        }
                    }   // end for start
                }   // end for diff
            }
        }
    }


    com.kushalarora.compositionalLM.options.Options op;

    public StanfordGrammar(BinaryGrammar bg, UnaryGrammar ug, Lexicon lex,
                           com.kushalarora.compositionalLM.options.Options op,
                           Options defaultOp,   // TODO:: Deprecate this
                           Index<String> stateIndex, Index<String> wordIndex,
                           Index<String> tagIndex) {
        super(bg, ug, lex, defaultOp, stateIndex, wordIndex, tagIndex);
        this.op = op;
    }


    /**
     * Compute inside and outside score for the sentence.
     * Also computes span and span split score we need.
     *
     * @param sentence Sentence being processed.
     */
    public IInsideOutsideScores computeInsideOutsideProb(List<Word> sentence) {
        val insideOutsideScore = new StanfordInsideOutsideScore(sentence);

        insideOutsideScore.initializeScoreArrays();

        insideOutsideScore.doLexScores();

        log.debug("Starting inside score computation");
        insideOutsideScore.doInsideScores();


        log.debug("Start outside score computation");
        insideOutsideScore.doOutsideScores();

        log.debug("Start mu score computation");
        insideOutsideScore.computeMuSpanScore();

        return insideOutsideScore;
    }


    public IInsideOutsideScores getInsideOutsideObject(List<Word> sentence) {
        return new StanfordInsideOutsideScore(sentence);
    }

    public int getNumStates() {
        return numStates;
    }

    public int getVocabSize() {
        return wordIndex.size();
    }
}

