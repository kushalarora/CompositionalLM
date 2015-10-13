package com.kushalarora.compositionalLM.lang;

import com.kushalarora.compositionalLM.options.Options;
import edu.stanford.nlp.ling.HasContext;
import edu.stanford.nlp.parser.lexparser.*;
import edu.stanford.nlp.util.Index;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang.StringUtils;

import java.util.*;

import static com.kushalarora.compositionalLM.utils.ObjectSizeFetcher.getSize;
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
    public void doLexScores(AbstractInsideOutsideScore score) {
        StanfordInsideOutsideScore s = (StanfordInsideOutsideScore) score;

        int length = s.getLength();
        Sentence sentence = s.getSentence();
        s.words = new int[length];
        boolean[][] tags = new boolean[length][numStates];

        for (int i = 0; i < length; i++) {
         //    String word = sentence.get(i).word();
            s.words[i] = sentence.get(i).getIndex();
        }

        for (int start = 0; start < length; start++) {
            int word = s.words[start];
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

                    s.addToScore(s.iScore, tot, start, end, state);
                    // iScore_start_end[state] += tot;

                    // in case of leaf nodes there is no node, hence
                    // we keeping the value of split at start
                    s.addToScore(s.iSplitSpanStateScore, tot, start, end, start, state);

                    s.addToScore(s.iSpanSplitScore, tot, start, end, start);

                    s.addToScore(s.iSpanScore, tot, start, end);
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
                    if (isTag[state] && s.iScore.getAsDouble(start, end, state) == Double.NEGATIVE_INFINITY) {

                        double lexScore = lex.score(new IntTaggedWord(word,
                                        tagIndex.indexOf(stateIndex.get(state))),
                                start, wordIndex.get(word), wordContextStr);
                        if (lexScore > Double.NEGATIVE_INFINITY) {
                            double tot = exp(lexScore);

                            s.addToScore(s.iScore, tot, start, end, state);
                            // in case of leaf nodes there is no node, hence
                            // we keeping the value of split at start
                            s.addToScore(s.iSplitSpanStateScore, tot, start, end, start, state);

                            s.addToScore(s.iSpanSplitScore, tot, start, end, start);

                            s.addToScore(s.iSpanScore, tot, start, end);

//                                iSplitSpanStateScore[start][end][start][state] += tot;
//                                iSpanSplitScore[start][end][start] += tot;
//                                iSpanScore[start][end] += tot;
                        }
                    }
                }
            } // end if ! assignedSomeTag

            // Apply unary rules
            for (int state = 0; state < numStates; state++) {

                double iS = s.iScore.getAsDouble(start, end, state);
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
                    s.addToScore(s.iScore, tot, start, end, parentState);

                    // in case of leaf nodes there is no node, hence
                    // we keeping the value of split at start
                    s.addToScore(s.iSplitSpanStateScore, tot, start, end, start, parentState);

                    s.addToScore(s.iSpanSplitScore, tot, start, end, start);

                    s.addToScore(s.iSpanScore, tot, start, end);

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
    public void doInsideScores(AbstractInsideOutsideScore score) {
        StanfordInsideOutsideScore s = (StanfordInsideOutsideScore) score;

        for (int diff = 2; diff <= s.length; diff++) {
            // usually stop one short because boundary symbol only combines
            // with whole sentence span. So for 3 word sentence + boundary = 4,
            // length == 4, and do [0,2], [1,3]; [0,3]; [0,4]
            for (int start = 0; start < ((diff == s.length) ? 1 : s.length - diff); start++) {
                doInsideChartCell(s, start, start + diff);
            } // for start
        } // for diff (i.e., span)
    } // end doInsideScores()


    /**
     * Compute inside, inside span, inside span split Score for span (start,end).
     *
     * @param start start index of span
     * @param end   end index of span
     */
    private void doInsideChartCell(StanfordInsideOutsideScore s, final int start, final int end) {
        log.debug("Doing iScore for span {} - {}", start, end);
        boolean[][] stateSplit = new boolean[numStates][s.length];
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
                    double lS = s.getScore(s.iScore, start, split, leftState);

                    if (lS == 0f) {
                        continue;
                    }
                    lS = log(lS);

//                        double rS = iScore[split][end][rightState];
                    double rS = s.getScore(s.iScore, split, end, rightState);
                    if (rS == 0f) {
                        continue;
                    }
                    rS = log(rS);

                    binaryRuleSet.add(rule);

                    stateSplit[parentState][split] = true;

                    double tot = exp(pS + lS + rS);

                    // in left child
//                        iScore[start][end][parentState] += tot;
                    s.addToScore(s.iScore, tot, start, end, parentState);

                    // Marginalizing over parentState and retaining
                    // split index
//                        iSpanSplitScore[start][end][split] += tot;
                    s.addToScore(s.iSpanSplitScore, tot, start, end, split);

//                        iSplitSpanStateScore[start][end][split][parentState] += tot;
                    s.addToScore(s.iSplitSpanStateScore, tot, start, end, split, parentState);

//                       iSpanScore[start][end] += tot;
                    s.addToScore(s.iSpanScore, tot, start, end);
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
                    double lS = s.getScore(s.iScore, start, split, leftState);
                    if (lS == 0f) {
                        continue;
                    }
                    lS = log(lS);

//                        double rS = iScore[split][end][rightState];
                    double rS = s.getScore(s.iScore, split, end, rightState);
                    if (rS == 0f) {
                        continue;
                    }
                    rS = log(rS);

                    binaryRuleSet.add(rule);

                    stateSplit[parentState][split] = true;

                    double tot = exp(pS + lS + rS);
                    // right child
//                        iScore[start][end][parentState] += tot;
                    s.addToScore(s.iScore, tot, start, end, parentState);

                    // Marginalizing over parentState and retaining
                    // split index
                    //iSpanSplitScore[start][end][split] += tot;
                    s.addToScore(s.iSpanSplitScore, tot, start, end, split);

//                        iSplitSpanStateScore[start][end][split][parentState] += tot;
                    s.addToScore(s.iSplitSpanStateScore, tot, start, end, split, parentState);

//                       iSpanScore[start][end] += tot;
                    s.addToScore(s.iSpanScore, tot, start, end);

                } // for split point
            } // end for rightRules
        }

        // do unary rules -- one could promote this loop and put start inside
        for (int state = 0; state < numStates; state++) {

//                double iS = iScore[start][end][state];
            double iS = s.getScore(s.iScore, start, end, state);
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
                s.addToScore(s.iScore, tot, start, end, parentState);


                // We are marginalizing over all states and this state
                // spans (start,end) and should be marginalized for all
                // splits
                for (int split = start + 1; split < end; split++) {
                    if (stateSplit[state][split]) {
//                            iSpanSplitScore[start][end][split] += tot;
                        s.addToScore(s.iSpanSplitScore, tot, start, end, split);

//                            iSplitSpanStateScore[start][end][split][parentState] += tot;
                        s.addToScore(s.iSplitSpanStateScore, tot, start, end, split, parentState);

//                            iSpanScore[start][end] += tot;
                        s.addToScore(s.iSpanScore, tot, start, end);
                    }
                }

            } // for UnaryRule r
        } // for unary rules
    }

    /**
     * Populate outside score related arrays.
     */
    public void doOutsideScores2(AbstractInsideOutsideScore score) {
        StanfordInsideOutsideScore s = (StanfordInsideOutsideScore) score;
        int initialParentIdx = s.length;
        int initialStart = 0;
        int initialEnd = s.length;
        int startSymbol = stateIndex.indexOf(goalStr);
//            oScore[initialStart][initialEnd][startSymbol] = 1.0f;
//            oSpanWParentScore[initialStart][initialEnd][initialParentIdx] = 1.0f;
//            oSpanStateScoreWParent[initialStart][initialEnd][initialParentIdx][startSymbol] = 1.0f;

        s.setScore(s.oScore, 1.0f,
                initialStart, initialEnd, startSymbol);

        s.setScore(s.oSpanWParentScore, 1.0f,
                initialStart, initialEnd, initialParentIdx);

        s.setScore(s.oSpanStateScoreWParent, 1.0f,
                initialStart, initialEnd, initialParentIdx, startSymbol);


        for (int diff = s.length; diff >= 1; diff--) {
            for (int start = 0; start + diff <= s.length; start++) {
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
                    double oS = s.getScore(s.oScore, start, end, parentState);

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

                        s.addToScore(s.oScore, tot, start, end, childState);
                        s.addToScore(s.oSpanWParentScore, tot, start, end, parent);
                        s.addToScore(s.oSpanStateScoreWParent, tot, start, end, parent, childState);
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
                        double oS = s.getScore(s.oScore, start, end, parentState);
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
                            double rS = s.getScore(s.iScore, split, end, rightState);
                            if (rS > 0f) {
                                rS = log(rS);

                                double totR = exp(pS + rS + oS);

///                                    oScore[lStart][lEnd][leftState] += totR;
//                                    oSpanWParentScore[lStart][lEnd][lParent] += totR;
//                                    oSpanStateScoreWParent[lStart][lEnd][lParent][leftState] += totR;

                                s.addToScore(s.oScore, totR, lStart, lEnd, leftState);
                                s.addToScore(s.oSpanWParentScore, totR, lStart, lEnd, lParent);
                                s.addToScore(s.oSpanStateScoreWParent, totR, lStart, lEnd, lParent, leftState);
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
                        double oS = s.getScore(s.oScore, start, end, parentState);
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
                            double lS = s.getScore(s.iScore, start, split, leftState);
                            if (lS > 0f) {
                                lS = log(lS);
                                double totL = exp(pS + lS + oS);


//                                    oScore[rStart][rEnd][rightState] += totL;
//                                    oSpanWParentScore[rStart][rEnd][rParent] += totL;
//                                    oSpanStateScoreWParent[rStart][rEnd][rParent][rightState] += totL;
                                s.addToScore(s.oScore, totL, rStart, rEnd, rightState);
                                s.addToScore(s.oSpanWParentScore, totL, rStart, rEnd, rParent);
                                s.addToScore(s.oSpanStateScoreWParent, totL, rStart, rEnd, rParent, rightState);
                            }
                        }   // end for split
                    }   // end for binary rules iter
                }   // end for right state
            }   // end for start
        }   // end for end
    }   // end doOutsideScores


    public void doOutsideScores(AbstractInsideOutsideScore score) {
        StanfordInsideOutsideScore s = (StanfordInsideOutsideScore) score;

        int initialParentIdx = s.length;
        int initialStart = 0;
        int initialEnd = s.length;
        int startSymbol = stateIndex.indexOf(goalStr);
//            oScore[initialStart][initialEnd][startSymbol] = 1.0f;
//            oSpanWParentScore[initialStart][initialEnd][initialParentIdx] = 1.0f;
//            oSpanStateScoreWParent[initialStart][initialEnd][initialParentIdx][startSymbol] = 1.0f;

        s.setScore(s.oScore, 1.0f,
                initialStart, initialEnd, startSymbol);

        s.setScore(s.oSpanWParentScore, 1.0f,
                initialStart, initialEnd, initialParentIdx);

        s.setScore(s.oSpanStateScoreWParent, 1.0f,
                initialStart, initialEnd, initialParentIdx, startSymbol);


        for (int diff = s.length; diff >= 1; diff--) {
            for (int start = 0; start + diff <= s.length; start++) {
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
                    double oS = s.getScore(s.oScore, start, end, parentState);

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

                        s.addToScore(s.oScore, tot, start, end, childState);
                        s.addToScore(s.oSpanWParentScore, tot, start, end, parent);
                        s.addToScore(s.oSpanStateScoreWParent, tot, start, end, parent, childState);
                    }   // end for unary rule iter
                }   // end for parentState

                // do binaries
                for (int parentState = 0; parentState < numStates; parentState++) {
                    // if current parentState's outside score is zero,
                    // child's would be zero as well
                    double oS = s.getScore(s.oScore, start, end, parentState);
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

                            double rS = s.getScore(s.iScore, split, end, rightState);
                            if (rS > 0f) {
                                rS = log(rS);

                                double totR = exp(pS + rS + oS);

//                                    oScore[lStart][lEnd][leftState] += totR;
//                                    oSpanWParentScore[lStart][lEnd][lParent] += totR;
//                                    oSpanStateScoreWParent[lStart][lEnd][lParent][leftState] += totR;

                                s.addToScore(s.oScore, totR, lStart, lEnd, leftState);
                                s.addToScore(s.oSpanWParentScore, totR, lStart, lEnd, lParent);
                                s.addToScore(s.oSpanStateScoreWParent, totR, lStart, lEnd, lParent, leftState);
                            } // end if rs > 0


                            // If iScore of the left span is zero, so is the
                            // oScore of left span
                            double lS = s.getScore(s.iScore, start, split, leftState);
                            if (lS > 0f) {
                                lS = log(lS);
                                double totL = exp(pS + lS + oS);

//                                    oScore[rStart][rEnd][rightState] += totL;
//                                    oSpanWParentScore[rStart][rEnd][rParent] += totL;
//                                    oSpanStateScoreWParent[rStart][rEnd][rParent][rightState] += totL;
                                s.addToScore(s.oScore, totL, rStart, rEnd, rightState);
                                s.addToScore(s.oSpanWParentScore, totL, rStart, rEnd, rParent);
                                s.addToScore(s.oSpanStateScoreWParent, totL, rStart, rEnd, rParent, rightState);

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
    public void doMuScore(AbstractInsideOutsideScore score) {
        StanfordInsideOutsideScore s = (StanfordInsideOutsideScore) score;


        // Handle lead node case.
        // There is no split here and span value
        // is stored at start
        for (int start = 0; start < s.length; start++) {
            int end = start + 1;
            int split = start;
            log.debug("Doing muScore for span {} - {}", start, end);
            for (int state = 0; state < numStates; state++) {

                // If iScore or oScore is zero, the mu score is zero
                double iS = s.getScore(s.iScore, start, end, state);
                if (iS == 0) {
                    continue;
                }
                iS = log(iS);

                double oS = s.getScore(s.oScore, start, end, state);

                if (oS == 0f) {
                    continue;
                }
                oS = log(oS);


                double tot;
                tot = exp(iS + oS);
                s.addToScore(s.muScore, tot, start, end, state);


                log.debug("muScore[{}][{}][{}] = {}",
                        start, end, state, tot);

                // Will surely reach here if this is non zero
                // as iScore is nothing but marginalization over split
                // If this is zero, then this split with this state
                // din't occur in grammar
                double iSplitSpanStateScore = s.getScore(
                        s.iSplitSpanStateScore, start, end, split, state);

                if (iSplitSpanStateScore == 0) {
                    continue;
                }
                iSplitSpanStateScore = log(iSplitSpanStateScore);

                // Takes care of parents of span (parentBegin, end)
                for (int parentBegin = 0; parentBegin < start; parentBegin++) {
                    double oScoreWParent = s.getScore(
                            s.oSpanStateScoreWParent, start, end, parentBegin, state);

                    // If this is zero then parent  (parentBegin, end)
                    // were never expanded to child with this state spanning
                    // (start, end)
                    if (oScoreWParent == 0f) {
                        continue;
                    }
                    oScoreWParent = log(oScoreWParent);
                    tot = exp(oScoreWParent + iSplitSpanStateScore);
                    s.addToScore(
                            s.muSpanSplitScoreWParent, tot, start, end, split, parentBegin);

                }

                // Takes care of the parents with span (start, parentEnd)
                for (int parentEnd = end; parentEnd <= s.length; parentEnd++) {
                    double oScoreWParent = s.getScore(
                            s.oSpanStateScoreWParent, start, end, parentEnd, state);

                    // If this is zero then parent  (start, parentEnd)
                    // were never expanded to child with this state spanning
                    // (start, end)
                    if (oScoreWParent == 0f) {
                        continue;
                    }
                    oScoreWParent = log(oScoreWParent);
                    tot = exp(oScoreWParent + iSplitSpanStateScore);

                    s.addToScore(s.muSpanSplitScoreWParent, tot, start, end, split, parentEnd);
                }
            }
        }

        // Handle cases for diff = 2 and above
        for (int diff = 2; diff <= s.length; diff++) {
            for (int start = 0; start + diff <= s.length; start++) {
                int end = start + diff;
                log.debug("Doing muScore for span {} - {}", start, end);
                for (int state = 0; state < numStates; state++) {

                    // If iScore or oScore is zero, the mu score is zero
                    double oS = s.getScore(s.oScore, start, end, state);
                    if (oS == 0) {
                        continue;
                    }
                    oS = log(oS);

                    double iS = s.getScore(s.iScore, start, end, state);
                    if (iS == 0) {
                        continue;
                    }
                    iS = log(iS);

                    double tot;

                    tot = exp(oS + iS);
                    s.addToScore(s.muScore, tot, start, end, state);

                    log.debug("muScore[{}][{}][{}] = {}",
                            start, end, state, tot);
                    // This handles the leaf nodes with no split
                    for (int split = start + 1; split < end; split++) {
                        // Marginalizing over both split and parent state
                        double iSplitSpanStateScore = s.getScore(
                                s.iSplitSpanStateScore, start, end, split, state);

                        if (iSplitSpanStateScore == 0) {
                            continue;
                        }
                        iSplitSpanStateScore = log(iSplitSpanStateScore);

                        // Takes care of parents of span (parentBegin, end)
                        for (int parentBegin = 0; parentBegin < start; parentBegin++) {
                            double oScoreWParent = s.getScore(
                                    s.oSpanStateScoreWParent, start, end, parentBegin, state);

                            // If this is zero then parent  (parentBegin, end)
                            // were never expanded to child with this state spanning
                            // (start, end)
                            if (oScoreWParent == 0f) {
                                continue;
                            }
                            oScoreWParent = log(oScoreWParent);

                            tot = exp(oScoreWParent + iSplitSpanStateScore);
                            s.addToScore(s.muSpanSplitScoreWParent, tot, start, end, split, parentBegin);
                            log.debug("muSpanSplitScoreWParent[{}][{}][{}][{}] = {}",
                                    start, end, split, parentBegin, tot);
                        }

                        // Takes care of the parents with span (start, parentEnd)
                        for (int parentEnd = end; parentEnd <= s.length; parentEnd++) {
                            double oStateScoreWParent = s.getScore(s.oSpanStateScoreWParent, start, end, parentEnd, state);
                            // If this is zero then parent  (start, parentEnd)
                            // were never expanded to child with this state spanning
                            // (start, end)
                            if (oStateScoreWParent == 0f) {
                                continue;
                            }
                            oStateScoreWParent = log(oStateScoreWParent);

                            tot = exp(oStateScoreWParent + iSplitSpanStateScore);
                            s.addToScore(s.muSpanSplitScoreWParent, tot, start, end, split, parentEnd);
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
    public void computeInsideOutsideProb(AbstractInsideOutsideScore score) {
        StanfordInsideOutsideScore s = (StanfordInsideOutsideScore) score;

        int idx = s.sentence.getIndex();
        int sz = s.sentence.size();

        log.info("Starting inside score computation:{}::{}", idx, sz);
        doLexScores(s);
        doInsideScores(s);
        log.info("Computed inside score computation:{}::{}", idx, sz);


        log.info("Start outside score computation:{}::{}", idx, sz);
        doOutsideScores2(s);
        log.info("Computed outside score computation:{}::{}", idx, sz);


        log.info("Start mu score computation:{}::{}", idx, sz);
        doMuScore(s);
        log.info("Computed mu score computation:{}::{}", idx, sz);

        s.clearTempArrays();
        s.clearNonSpanArrays();


        if (op.debug) {
            log.info("Memory Size StanfordIOScore: {}:: {}\n" +
                            "\t {} => {} MB\n" +
                            "\t {} => {} MB\n" +
                            "\t {} => {} MB\n" +
                            "\t {} => {} MB\n" +
                            "total => {} MB",
                    idx, sz,
                    "iSpanScore", getSize(s.iSpanScore),
                    "iSpanSplitScore", getSize(s.iSpanSplitScore),
                    "oSpanWParentScore", getSize(s.oSpanWParentScore),
                    "muSpanSplitScoreWParent", getSize(s.muSpanSplitScoreWParent),
                    getSize(s));
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
        StanfordInsideOutsideScore score = new StanfordInsideOutsideScore(sentence, numStates);
        computeInsideOutsideProb(score);
        return score;
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
            signature = str.toLowerCase();
        }

        if (!wordIndex.contains(signature)) {
            signature = str.toLowerCase(Locale.ENGLISH);
            if (!wordIndex.contains(signature)) {
                signature = str.toUpperCase();
                if (!wordIndex.contains(signature)) {
                    signature = StringUtils.capitalize(str);
                    if (!wordIndex.contains(signature)) {
                        signature = lex.getUnknownWordModel().getSignature(str, loc);
                    }
                }
            }
        }

        index = wordIndex.indexOf(signature);


        // If we aren't able to find signature
        // lets use UNK
        if (index == -1) {
            signature = "UNK";
            index = wordIndex.indexOf(signature);
        }
        return new Word(str, index, signature);
    }
}