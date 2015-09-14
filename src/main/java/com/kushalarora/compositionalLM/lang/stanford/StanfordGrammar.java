package com.kushalarora.compositionalLM.lang.stanford;

import com.kushalarora.compositionalLM.lang.AbstractGrammar;
import com.kushalarora.compositionalLM.lang.AbstractInsideOutsideScore;
import com.kushalarora.compositionalLM.lang.Sentence;
import com.kushalarora.compositionalLM.lang.Word;
import com.kushalarora.compositionalLM.options.Options;
import com.kushalarora.compositionalLM.utils.ObjectSizeFetcher;
import edu.stanford.nlp.ling.HasContext;
import edu.stanford.nlp.parser.lexparser.*;
import edu.stanford.nlp.util.Index;
import lombok.extern.slf4j.Slf4j;
import org.ujmp.core.SparseMatrix;

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

// TODO:: Avoid repeatition in all functions while handling
// unary and binary rule cases by creating sub routines
// of the common code with arguments start, end, split

@Slf4j
public class StanfordGrammar extends AbstractGrammar {
    private final LexicalizedParser model;

    /**
     * Created by karora on 6/24/15.
     */

    public class StanfordInsideOutsideScore extends AbstractInsideOutsideScore {

        private transient SparseMatrix iSplitSpanStateScore;
        private transient SparseMatrix oSpanStateScoreWParent;
        protected int[] words;  // words of sentence being parsed as word Numberer ints

        public StanfordInsideOutsideScore(Sentence sentence) {
            super(sentence);
        }

        /**
         * Deallocate all the arrays.
         */
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

            // [start][end][parent]
            oSpanWParentScore = null;

            // [start][end][parent][state]
            oSpanStateScoreWParent = null;

            // [start][end][state]
            muScore = null;

            // [start][end][split][parent]
            muSpanSplitScoreWParent = null;

            arraySize = 0;
        }

        private void clearTempArrays() {
            iSplitSpanStateScore = null;
            oSpanStateScoreWParent = null;
        }

        /**
         * Conditionally create inside, outside, [narrow|wide][R|L]Extent arrays,
         * iSpanSplitScore. If unable to create array try restore the current size.
         */

        // Kushal::Method private in base class
        public void considerCreatingArrays() {
            createArrays(length);
            log.info("Created arrays of size " + length);
        }

        /**
         * Create inside, outside, [narrow|wide][R|L]Extent arrays,
         * iSpanSplitScore.
         */
        // Kushal::Added initialization code for span scores
        private void createArrays(int length) {
            // zero out some stuff first in case we recently
            // ran out of memory and are reallocating
            clearArrays();
            log.info("Starting array allocation");
/*            iScore = new double[length][length + 1][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    iScore[start][end] = new double[numStates];
                }
            }*/
            iScore = SparseMatrix.Factory.zeros(length, length + 1, numStates);

/*
            oScore = new double[length][length + 1][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    oScore[start][end] = new double[numStates];
                }
            }*/
            oScore = SparseMatrix.Factory.zeros(length, length + 1, numStates);

//            iSpanScore = new double[length][length + 1];
            iSpanScore = SparseMatrix.Factory.zeros(length, length + 1);

/*            iSpanSplitScore = new double[length][length + 1][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    // splits
                    iSpanSplitScore[start][end] = new double[length];
                }
            }*/
            iSpanSplitScore = SparseMatrix.Factory.zeros(length, length + 1, length);

/*            oSpanWParentScore = new double[length][length + 1][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    // parents
                    oSpanWParentScore[start][end] = new double[length + 1];
                }
            }*/
            oSpanWParentScore = SparseMatrix.Factory.zeros(length, length + 1, length + 1);

/*
            oSpanStateScoreWParent = new double[length][length + 1][][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    // parents
                    oSpanStateScoreWParent[start][end] = new double[length + 1][];
                    for (int parent = 0; parent < start; parent++) {
                        // states
                        oSpanStateScoreWParent[start][end][parent] = new double[numStates];
                    }

                    for (int parent = end; parent <= length; parent++) {
                        // states
                        oSpanStateScoreWParent[start][end][parent] = new double[numStates];
                    }
                }
            }*/
            oSpanStateScoreWParent = SparseMatrix.Factory.zeros(length, length + 1, length + 1, numStates);


/*
            iSplitSpanStateScore = new double[length][length + 1][][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    iSplitSpanStateScore[start][end] = new double[length][];
                    for (int split = start; split < end; split++) {
                        // states
                        iSplitSpanStateScore[start][end][split] = new double[numStates];
                    }
                }
            }*/
            iSplitSpanStateScore = SparseMatrix.Factory.zeros(length, length + 1, length, numStates);

/*
            muScore = new double[length][length + 1][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    muScore[start][end] = new double[numStates];
                }
            }*/
            muScore = SparseMatrix.Factory.zeros(length, length + 1, numStates);

/*
            muSpanSplitScoreWParent = new double[length][length + 1][][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    // splits
                    muSpanSplitScoreWParent[start][end] = new double[length][];
                    for (int split = start; split < end; split++) {
                        // parents
                        muSpanSplitScoreWParent[start][end][split] = new double[length + 1];
                    }
                }
            }*/
            muSpanSplitScoreWParent = SparseMatrix.Factory.zeros(length, length + 1, length, length + 1);

            log.info("Finished allocating arrays of length {}", length);
        }

        @Override
        public void initializeScoreArrays() {
/*
            log.info("Intializing Inside Outside Arrays");
            if (length > arraySize) {
                considerCreatingArrays();
            }


            log.debug("Initializing arrays with 0f");
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    // fill states
                    Arrays.fill(iScore[start][end], 0f);
                    Arrays.fill(oScore[start][end], 0f);

                    iSpanScore[start][end] = 0f;

                    // fill splits (start, end)
                    Arrays.fill(iSpanSplitScore[start][end], 0f);

                    // fill splits (start, end)
                    Arrays.fill(oSpanWParentScore[start][end], 0f);

                    for (int split = start + 1; split < end; split++) {
                        // fill state
                        Arrays.fill(iSplitSpanStateScore[start][end][split], 0f);
                    }

                    for (int split = start + 1; split < end; split++) {
                        // fill parent
                        Arrays.fill(muSpanSplitScoreWParent[start][end][split], 0f);
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
                }
            }*/
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
            boolean[][] tags = new boolean[length][numStates];

            for (int i = 0; i < length; i++) {
                String s = sentence.get(i).word();
                words[i] = wordIndex.addToIndex(s);
            }

            for (int start = 0; start < length; start++) {
                int word = words[start];
                int end = start + 1;
                Arrays.fill(tags[start], false);
                log.debug("Doing lex score lookup for index {}", start);

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
                    double lexScore = lex.score(tagging, start,
                            wordIndex.get(tagging.word), wordContextStr);

                    if (lexScore > Double.NEGATIVE_INFINITY) {
                        assignedSomeTag = true;
                        double tot = exp(lexScore);

                        addToScore(iScore, tot, start, end, state);
                        // iScore_start_end[state] += tot;

                        // in case of leaf nodes there is no node, hence
                        // we keeping the value of split at start
                        addToScore(iSplitSpanStateScore,tot,start, end, start, state);

                        addToScore(iSpanSplitScore, tot, start, end, start);

                        addToScore(iSpanScore, tot, start, end);
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
                        if (isTag[state] && iScore.getAsDouble(start, end, state) == Double.NEGATIVE_INFINITY) {

                            double lexScore = lex.score(new IntTaggedWord(word,
                                            tagIndex.indexOf(stateIndex.get(state))),
                                    start, wordIndex.get(word), wordContextStr);
                            if (lexScore > Double.NEGATIVE_INFINITY) {
                                double tot = exp(lexScore);

                                addToScore(iScore, tot, start, end, state);
                                // in case of leaf nodes there is no node, hence
                                // we keeping the value of split at start
                                addToScore(iSplitSpanStateScore, tot, start, end, start, state);

                                addToScore(iSpanSplitScore, tot, start, end, start);

                                addToScore(iSpanScore, tot, start, end);

//                                iSplitSpanStateScore[start][end][start][state] += tot;
//                                iSpanSplitScore[start][end][start] += tot;
//                                iSpanScore[start][end] += tot;
                            }
                        }
                    }
                } // end if ! assignedSomeTag

                // Apply unary rules
                for (int state = 0; state < numStates; state++) {

                    double iS = iScore.getAsDouble(start, end, state);
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

                        // iScore_start_end[parentState] += tot;
                       addToScore(iScore, tot, start, end, parentState);

                        // in case of leaf nodes there is no node, hence
                        // we keeping the value of split at start
                        addToScore(iSplitSpanStateScore, tot, start, end, start, parentState);

                        addToScore(iSpanSplitScore, tot, start, end, start);

                        addToScore(iSpanScore, tot, start, end);

//                        iSplitSpanStateScore[start][end][start][parentState] += tot;
//                        iSpanSplitScore[start][end][start] += tot;
//                        iSpanScore[start][end] += tot;

                    }   // end for unary rules
                }   // end for state for unary rules

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
            log.debug("Doing iScore for span {} - {}", start, end);
            boolean[][] stateSplit = new boolean[numStates][length];
            Set<BinaryRule> binaryRuleSet = new HashSet<BinaryRule>();

            for (int leftState = 0; leftState < numStates; leftState++) {
                BinaryRule[] leftRules = bg.splitRulesWithLC(leftState);
                for (BinaryRule rule : leftRules) {

                    int rightState = rule.rightChild;
                    int parentState = rule.parent;

                    // This binary split might be able to cover the span depending upon children's coverage.
                    float pS = rule.score;


                    // calculate iScore for state by summing over split
                    // and iSpanSplitScore by summing over states
                    // Stops one short of end as last span should be
                    // (start, end - 1), (end - 1, end)
                    for (int split = start + 1; split < end; split++) {

//                        double lS = iScore[start][split][leftState];
                        double lS = getScore(iScore, start, split, leftState);

                        if (lS == 0f) {
                            continue;
                        }
                        lS = log(lS);

//                        double rS = iScore[split][end][rightState];
                        double rS = getScore(iScore, split, end, rightState);
                        if (rS == 0f) {
                            continue;
                        }
                        rS = log(rS);

                        binaryRuleSet.add(rule);

                        stateSplit[parentState][split] = true;

                        double tot = exp(pS + lS + rS);

                        // in left child
//                        iScore[start][end][parentState] += tot;
                        addToScore(iScore, tot, start, end, parentState);

                        // Marginalizing over parentState and retaining
                        // split index
//                        iSpanSplitScore[start][end][split] += tot;
                        addToScore(iSpanSplitScore, tot, start, end, split);

//                        iSplitSpanStateScore[start][end][split][parentState] += tot;
                        addToScore(iSplitSpanStateScore, tot, start, end, split, parentState);

//                       iSpanScore[start][end] += tot;
                        addToScore(iSpanScore, tot, start, end);
                    } // for split point
                } // end for leftRules
            }

            for (int rightState = 0; rightState < numStates; rightState++) {
                BinaryRule[] rightRules = bg.splitRulesWithRC(rightState);
                for (BinaryRule rule : rightRules) {

                    // Rule already processed by left state loop
                    if (binaryRuleSet.contains(rule)) {
                        log.debug("Rule {} already processed by left child loop.Skipping", rule);
                        continue;
                    }

                    int leftState = rule.leftChild;
                    int parentState = rule.parent;

                    // This binary split might be able to cover the span depending upon children's coverage.
                    float pS = rule.score;


                    // calculate iScore for state by summing over split
                    // and iSpanSplitScore by summing over states
                    // Stops one short of end as last span should be
                    // (start, end - 1), (end - 1, end)
                    for (int split = start + 1; split < end; split++) {

//                        double lS = iScore[start][split][leftState];
                        double lS = getScore(iScore, start, split, leftState);
                        if (lS == 0f) {
                            continue;
                        }
                        lS = log(lS);

//                        double rS = iScore[split][end][rightState];
                        double rS = getScore(iScore, split, end, rightState);
                        if (rS == 0f) {
                            continue;
                        }
                        rS = log(rS);

                        binaryRuleSet.add(rule);

                        stateSplit[parentState][split] = true;

                        double tot = exp(pS + lS + rS);
                        // right child
//                        iScore[start][end][parentState] += tot;
                        addToScore(iScore, tot, start, end, parentState);

                        // Marginalizing over parentState and retaining
                        // split index
                        //iSpanSplitScore[start][end][split] += tot;
                        addToScore(iSpanSplitScore, tot, start, end, split);

//                        iSplitSpanStateScore[start][end][split][parentState] += tot;
                        addToScore(iSplitSpanStateScore, tot, start, end, split, parentState);

                        addToScore(iSpanScore, tot, start, end);

                    } // for split point
                } // end for rightRules
            }

            // do unary rules -- one could promote this loop and put start inside
            for (int state = 0; state < numStates; state++) {

//                double iS = iScore[start][end][state];
                double iS = getScore(iScore, start, end, state);
                if (iS == 0f) {
                    continue;
                }
                iS = log(iS);

                UnaryRule[] unaries = ug.closedRulesByChild(state);
                for (UnaryRule ur : unaries) {

                    int parentState = ur.parent;
                    float pS = ur.score;
                    double tot = exp(iS + pS);
//                    iScore[start][end][parentState] += tot;
                    addToScore(iScore, tot, start, end, parentState);


                    // We are marginalizing over all states and this state
                    // spans (start,end) and should be marginalized for all
                    // splits
                    for (int split = start + 1; split < end; split++) {
                        if (stateSplit[state][split]) {
//                            iSpanSplitScore[start][end][split] += tot;
                            addToScore(iSpanSplitScore, tot, start, end, split);

//                            iSplitSpanStateScore[start][end][split][parentState] += tot;
                            addToScore(iSplitSpanStateScore, tot, start, end, split, parentState);

//                            iSpanScore[start][end] += tot;
                            addToScore(iSpanScore, tot, start, end);
                        }
                    }

                } // for UnaryRule r
            } // for unary rules
        }

        /**
         * Populate outside score related arrays.
         */
        /*public void doOutsideScores2() {
            int initialParentIdx = length;
            int initialStart = 0;
            int initialEnd = length;
            int startSymbol = stateIndex.indexOf(goalStr);
            oScore[initialStart][initialEnd][startSymbol] = 1.0f;
            oSpanWParentScore[initialStart][initialEnd][initialParentIdx] = 1.0f;
            oSpanStateScoreWParent[initialStart][initialEnd][initialParentIdx][startSymbol] = 1.0f;

            for (int diff = length; diff >= 1; diff--) {
                for (int start = 0; start + diff <= length; start++) {
                    int end = start + diff;

                    log.debug("Doing oScore for span ({}, {})", start, end);
                    // do unaries
                    for (int parentState = 0; parentState < numStates; parentState++) {
                        // As this is unary rule and parent span is same as child,
                        // so consider the whole sentence to be parent ending with end
                        int parent = end;

                        // if current parentState's outside score is zero,
                        // child's would be zero as well
                        double oS = oScore[start][end][parentState];
                        if (oS == 0f) {
                            continue;
                        }
                        oS = log(oS);

                        UnaryRule[] rules = ug.closedRulesByParent(parentState);
                        for (UnaryRule ur : rules) {
                            double pS = ur.score;
                            int childState = ur.child;
                            double tot = exp(oS + pS);
                            log.debug("Adding unary rule {} to outside score for Start: {}, End: {}"
                                    , ur, start, end);

                            oScore[start][end][childState] += tot;
                            oSpanWParentScore[start][end][parent] += tot;
                            oSpanStateScoreWParent[start][end][parent][childState] += tot;
                        }   // end for unary rule iter
                    }   // end for parentState

                    // do binaries

                    // Outside score with left child not expanded
                    for (int leftState = 0; leftState < numStates; leftState++) {
                        // Left span starts at start and ends at split.
                        // The parent ends at end and is stored, the parent
                        // begins at start
                        int lStart = start;
                        int lParent = end;
                        BinaryRule[] rules = bg.splitRulesWithLC(leftState);
                        for (BinaryRule br : rules) {
                            int rightState = br.rightChild;
                            int parentState = br.parent;
                            // If paren't outside score is zero, so will be
                            // the child's
                            double oS = oScore[start][end][parentState];
                            if (oS == 0f) {
                                continue;
                            }
                            oS = log(oS);

                            double pS = br.score;


                            for (int split = start + 1; split < end; split++) {
                                // left span ends at end.
                                int lEnd = split;

                                // iScore of the right child.
                                // If the right child's iScore is zero, so
                                // will be the oScore of left child.
                                double rS = iScore[split][end][rightState];
                                if (rS > 0f) {
                                    rS = log(rS);

                                    log.debug("oScore[{}][{}][{}]= pS + oScore[{}][{}][{}]({}) + iScore[{}][{}][{}]" +
                                                    "({}) " +
                                                    "Rule" +
                                                    " {}",
                                            start, split, leftState,
                                            start, end, br.parent, oS,
                                            split, end, br.rightChild, rS,
                                            br);

                                    double totR = exp(pS + rS + oS);

                                    oScore[lStart][lEnd][leftState] += totR;
                                    oSpanWParentScore[lStart][lEnd][lParent] += totR;
                                    oSpanStateScoreWParent[lStart][lEnd][lParent][leftState] += totR;
                                }

                            }   // end for split
                        }   // end for binary rule iter
                    }   // end for leftState

                    // Outside score with right child not expanded
                    for (int rightState = 0; rightState < numStates; rightState++) {
                        // for right span, the span spans (split, end) with parents left endpoint
                        // stored in start, with right endpoint being end.
                        int rEnd = end;
                        int rParent = start;

                        BinaryRule[] rules = bg.splitRulesWithRC(rightState);
                        for (BinaryRule br : rules) {
                            int parentState = br.parent;
                            int leftState = br.leftChild;
                            //  if oScore of parent is zero, so is child's.
                            double oS = oScore[start][end][parentState];
                            if (oS == 0f) {
                                continue;
                            }
                            oS = log(oS);

                            double pS = br.score;

                            for (int split = start + 1; split < end; split++) {
                                // the left endpoint of  right span is split.
                                int rStart = split;

                                // If iScore of the left span is zero, so is the
                                // oScore of left span
                                double lS = iScore[start][split][leftState];
                                if (lS > 0f) {
                                    lS = log(lS);

                                    log.debug("oScore[{}][{}][{}]=oScore[{}][{}][{}]({}) + iScore[{}][{}][{}]({}) Rule {}",
                                            split, end, rightState,
                                            start, end, br.parent, oS,
                                            start, split, br.leftChild, lS,
                                            br);
                                    double totL = exp(pS + lS + oS);
                                    oScore[rStart][rEnd][rightState] += totL;
                                    oSpanWParentScore[rStart][rEnd][rParent] += totL;
                                    oSpanStateScoreWParent[rStart][rEnd][rParent][rightState] += totL;
                                }
                            }   // end for split
                        }   // end for binary rules iter
                    }   // end for right state
                }   // end for start
            }   // end for end
        }   // end doOutsideScores*/


        public void doOutsideScores() {
            int initialParentIdx = length;
            int initialStart = 0;
            int initialEnd = length;
            int startSymbol = stateIndex.indexOf(goalStr);
//            oScore[initialStart][initialEnd][startSymbol] = 1.0f;
//            oSpanWParentScore[initialStart][initialEnd][initialParentIdx] = 1.0f;
//            oSpanStateScoreWParent[initialStart][initialEnd][initialParentIdx][startSymbol] = 1.0f;

            setScore(oScore, 1.0f,
                    initialStart, initialEnd, startSymbol);

            setScore(oSpanWParentScore, 1.0f,
                    initialStart, initialEnd, initialParentIdx);

            setScore(oSpanStateScoreWParent, 1.0f,
                    initialStart, initialEnd, initialParentIdx, startSymbol);


            for (int diff = length; diff >= 1; diff--) {
                for (int start = 0; start + diff <= length; start++) {
                    int end = start + diff;

                    log.debug("Doing oScore for span ({}, {})", start, end);
                    // do unaries
                    for (int parentState = 0; parentState < numStates; parentState++) {
                        // As this is unary rule and parent span is same as child,
                        // so consider the whole sentence to be parent ending with end
                        int parent = end;

                        // if current parentState's outside score is zero,
                        // child's would be zero as well
//                        double oS = oScore[start][end][parentState];
                        double oS = getScore(oScore, start, end,parentState);

                        if (oS == 0f) {
                            continue;
                        }
                        oS = log(oS);

                        UnaryRule[] rules = ug.closedRulesByParent(parentState);
                        for (UnaryRule ur : rules) {
                            double pS = ur.score;
                            int childState = ur.child;
                            double tot = exp(oS + pS);
                            log.debug("Adding unary rule {} to outside score for Start: {}, End: {}"
                                    , ur, start, end);

//                            oScore[start][end][childState] += tot;
//                            oSpanWParentScore[start][end][parent] += tot;
//                            oSpanStateScoreWParent[start][end][parent][childState] += tot;

                            addToScore(oScore, tot, start, end, childState);
                            addToScore(oSpanWParentScore, tot, start, end, parent);
                            addToScore(oSpanStateScoreWParent, tot, start, end, parent, childState);
                        }   // end for unary rule iter
                    }   // end for parentState

                    // do binaries
                    for (int parentState = 0; parentState < numStates; parentState++) {
                        // if current parentState's outside score is zero,
                        // child's would be zero as well
                        double oS = getScore(oScore, start, end, parentState);
                        if (oS == 0f) {
                            continue;
                        }
                        oS = log(oS);

                        List<BinaryRule> rules = bg.ruleListByParent(parentState);
                        for (BinaryRule br : rules) {
                            int leftState = br.leftChild;
                            int rightState = br.rightChild;

                            double pS = br.score;

                            for (int split = start + 1; split < end; split++) {
                                int lStart = start, lEnd = split, lParent = end;
                                int rStart = split, rEnd = end, rParent = start;

                                double rS = getScore(iScore, split, end, rightState);
                                if (rS > 0f) {
                                    rS = log(rS);

                                    double totR = exp(pS + rS + oS);

//                                    oScore[lStart][lEnd][leftState] += totR;
//                                    oSpanWParentScore[lStart][lEnd][lParent] += totR;
//                                    oSpanStateScoreWParent[lStart][lEnd][lParent][leftState] += totR;

                                    addToScore(oScore, totR, lStart, lEnd, leftState);
                                    addToScore(oSpanWParentScore, totR, lStart, lEnd, lParent);
                                    addToScore(oSpanStateScoreWParent, totR, lStart, lEnd, lParent, leftState);
                                } // end if rs > 0


                                // If iScore of the left span is zero, so is the
                                // oScore of left span
                                double lS = getScore(iScore, start, split, leftState);
                                if (lS > 0f) {
                                    lS = log(lS);
                                    double totL = exp(pS + lS + oS);

//                                    oScore[rStart][rEnd][rightState] += totL;
//                                    oSpanWParentScore[rStart][rEnd][rParent] += totL;
//                                    oSpanStateScoreWParent[rStart][rEnd][rParent][rightState] += totL;
                                    addToScore(oScore, totL, rStart, rEnd, rightState);
                                    addToScore(oSpanWParentScore, totL, rStart, rEnd, rParent);
                                    addToScore(oSpanStateScoreWParent, totL, rStart, rEnd, rParent, rightState);

                                }   // end if ls > 0
                            }   // end for split
                        }
                    }   // end for parent state

                }   // end for start
            }   // end for end
        }   // end doOutsideScores

        /**
         * Populate mu score arrays
         */
        public void doMuScore() {

            // Handle lead node case.
            // There is no split here and span value
            // is stored at start
            for (int start = 0; start < length; start++) {
                int end = start + 1;
                int split = start;
                log.debug("Doing muScore for span {} - {}", start, end);
                for (int state = 0; state < numStates; state++) {

                    // If iScore or oScore is zero, the mu score is zero
                    double iS = getScore(iScore, start, end, state);
                    if (iS == 0) {
                        continue;
                    }
                    iS = log(iS);

                    double oS = getScore(oScore, start, end, state);

                    if (oS == 0f) {
                        continue;
                    }
                    oS = log(oS);


                    double tot;
                    tot = exp(iS + oS);
                    addToScore(muScore, tot, start, end, state);


                    log.debug("muScore[{}][{}][{}] = {}",
                            start, end, state, tot);

                    // Will surely reach here if this is non zero
                    // as iScore is nothing but marginalization over split
                    // If this is zero, then this split with this state
                    // din't occur in grammar
                    double iSplitSpanStateScore = getScore(
                            this.iSplitSpanStateScore, start, end, split, state);

                    if (iSplitSpanStateScore == 0) {
                        continue;
                    }
                    iSplitSpanStateScore = log(iSplitSpanStateScore);

                    // Takes care of parents of span (parentBegin, end)
                    for (int parentBegin = 0; parentBegin < start; parentBegin++) {
                        double oScoreWParent = getScore(
                                oSpanStateScoreWParent, start, end, parentBegin, state);

                        // If this is zero then parent  (parentBegin, end)
                        // were never expanded to child with this state spanning
                        // (start, end)
                        if (oScoreWParent == 0f) {
                            continue;
                        }
                        oScoreWParent = log(oScoreWParent);
                        tot = exp(oScoreWParent + iSplitSpanStateScore);
                        addToScore(
                                muSpanSplitScoreWParent, tot, start, end, split, parentBegin);

                    }

                    // Takes care of the parents with span (start, parentEnd)
                    for (int parentEnd = end; parentEnd <= length; parentEnd++) {
                        double oScoreWParent = getScore(
                                oSpanStateScoreWParent, start, end, parentEnd, state);

                        // If this is zero then parent  (start, parentEnd)
                        // were never expanded to child with this state spanning
                        // (start, end)
                        if (oScoreWParent == 0f) {
                            continue;
                        }
                        oScoreWParent = log(oScoreWParent);
                        tot = exp(oScoreWParent + iSplitSpanStateScore);

                        addToScore(muSpanSplitScoreWParent, tot, start, end, split, parentEnd);
                    }
                }
            }

            // Handle cases for diff = 2 and above
            for (int diff = 2; diff <= length; diff++) {
                for (int start = 0; start + diff <= length; start++) {
                    int end = start + diff;
                    log.debug("Doing muScore for span {} - {}", start, end);
                    for (int state = 0; state < numStates; state++) {

                        // If iScore or oScore is zero, the mu score is zero
                        double oS = getScore(oScore, start, end, state);
                        if (oS == 0) {
                            continue;
                        }
                        oS = log(oS);

                        double iS = getScore(iScore, start, end, state);
                        if (iS == 0) {
                            continue;
                        }
                        iS = log(iS);

                        double tot;

                        tot = exp(oS + iS);
                        addToScore(muScore, tot, start, end, state);

                        log.debug("muScore[{}][{}][{}] = {}",
                                start, end, state, tot);
                        // This handles the leaf nodes with no split
                        for (int split = start + 1; split < end; split++) {
                            // Marginalizing over both split and parent state
                            double iSplitSpanStateScore = getScore(
                                    this.iSplitSpanStateScore, start, end, split, state);

                            if (iSplitSpanStateScore == 0) {
                                continue;
                            }
                            iSplitSpanStateScore = log(iSplitSpanStateScore);

                            // Takes care of parents of span (parentBegin, end)
                            for (int parentBegin = 0; parentBegin < start; parentBegin++) {
                                double oScoreWParent = getScore(
                                        oSpanStateScoreWParent, start, end, parentBegin, state);

                                // If this is zero then parent  (parentBegin, end)
                                // were never expanded to child with this state spanning
                                // (start, end)
                                if (oScoreWParent == 0f) {
                                    continue;
                                }
                                oScoreWParent = log(oScoreWParent);

                                tot = exp(oScoreWParent + iSplitSpanStateScore);
                                addToScore(muSpanSplitScoreWParent, tot, start, end, split, parentBegin);
                                log.debug("muSpanSplitScoreWParent[{}][{}][{}][{}] = {}",
                                        start, end, split, parentBegin, tot);
                            }

                            // Takes care of the parents with span (start, parentEnd)
                            for (int parentEnd = end; parentEnd <= length; parentEnd++) {
                                double oStateScoreWParent = getScore(oSpanStateScoreWParent, start, end, parentEnd, state);
                                // If this is zero then parent  (start, parentEnd)
                                // were never expanded to child with this state spanning
                                // (start, end)
                                if (oStateScoreWParent == 0f) {
                                    continue;
                                }
                                oStateScoreWParent = log(oStateScoreWParent);

                                tot = exp(oStateScoreWParent + iSplitSpanStateScore);
                                addToScore(muSpanSplitScoreWParent, tot, start, end, split, parentEnd);
                                log.debug("muSpanSplitScoreWParent[{}][{}][{}][{}] = {}",
                                        start, end, split, parentEnd, tot);
                            }
                        }
                    }   // end for start
                }   // end for diff
            }
        }

        /**
         * Compute inside and outside score for the sentence.
         * Also computes span and span split score we need.
         */
        public void computeInsideOutsideProb() {

            considerCreatingArrays();
            initializeScoreArrays();

            log.info("Starting inside score computation:{}::{}", sentence.getIndex(), sentence.size());
            doLexScores();
            doInsideScores();
            log.info("Computed inside score computation:{}::{}", sentence.getIndex(), sentence.size());


            log.info("Start outside score computation:{}::{}", sentence.getIndex(), sentence.size());
            doOutsideScores();
            log.info("Computed outside score computation:{}::{}", sentence.getIndex(), sentence.size());

            log.info("Start mu score computation:{}::{}", sentence.getIndex(), sentence.size());
            doMuScore();
            log.info("Computed mu score computation:{}::{}", sentence.getIndex(), sentence.size());
            clearTempArrays();

            if (op.debug) {
                log.info("Memory Size StanfordIOScore: {}:: {} => {} MB",
                        sentence.getIndex(), sentence.size(),
                        ObjectSizeFetcher.getSize(this));
            }
        }

    }

    Options op;

    protected final String goalStr;
    protected final Index<String> stateIndex;
    protected final Index<String> wordIndex;
    protected final Index<String> tagIndex;

    protected final BinaryGrammar bg;
    protected final UnaryGrammar ug;
    protected final Lexicon lex;

    protected final int numStates;
    protected final boolean[] isTag;


    public StanfordGrammar(Options op,
                           LexicalizedParser model) {

        this.op = op;
        this.model = model;

        stateIndex = model.stateIndex;
        wordIndex = model.wordIndex;
        tagIndex = model.tagIndex;

        goalStr = model.treebankLanguagePack().startSymbol();
        bg = model.bg;
        ug = model.ug;
        lex = model.lex;
        numStates = model.stateIndex.size();

        isTag = new boolean[numStates];
        // tag index is smaller, so we fill by iterating over the tag index
        // rather than over the state index
        for (String tag : tagIndex.objectsList()) {
            int state = stateIndex.indexOf(tag);
            if (state < 0) {
                continue;
            }
            isTag[state] = true;
        }
    }

    public int getNumStates() {
        return numStates;
    }

    /**
     * Get inside outside score for sentence
     *
     * @param sentence Sentence for which inside outside score is to be calculated
     * @return InsideOutsideScore object
     */
    @Override
    public AbstractInsideOutsideScore getScore(Sentence sentence) {
        return new StanfordInsideOutsideScore(sentence);
    }

    /**
     * Return the vocabulary size
     *
     * @return vocab size
     */
    public int getVocabSize() {
        return wordIndex.size();
    }

    /**
     * Generate a word object from string by doing lex look up
     *
     * @param str String for the word
     * @param loc location of word in sentence
     * @return Returns a word object
     */
    public Word getToken(String str, int loc) {
        int index = -1;
        String signature = str;
        if (op.grammarOp.lowerCase) {
            signature = str.toLowerCase(Locale.ENGLISH);
        }

        if (!wordIndex.contains(signature)) {
            signature = lex.getUnknownWordModel().getSignature(str, loc);
        }

        index = wordIndex.indexOf(signature);

        if (index == -1) {
            // Ideally this should be
            // newline or something that
            // shouldn't show up at the end
            // In any case, the program would crash
            log.warn("word({}) or signature({}) not found in grammar",
                    str, signature);
        }
        return new Word(str, index, signature);
    }
}