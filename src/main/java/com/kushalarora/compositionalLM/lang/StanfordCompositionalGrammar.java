package com.kushalarora.compositionalLM.lang;

import com.google.common.base.Function;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.options.Options;
import com.kushalarora.compositionalLM.utils.Parallelizer;
import edu.stanford.nlp.parser.lexparser.*;
import edu.stanford.nlp.util.Index;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang.StringUtils;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.annotation.Nullable;
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
public class StanfordCompositionalGrammar extends AbstractGrammar {
    @Getter
    private final Model model;
    private final Options op;
    protected Parallelizer parallelizer;

    protected final String goalStr;
    protected final Index<String> stateIndex;
    protected final Index<String> wordIndex;
    protected final Index<String> tagIndex;

    protected final BinaryGrammar bg;
    protected final UnaryGrammar ug;
    protected final Lexicon lex;

    protected final int numStates;


    public StanfordCompositionalGrammar(Options op,
                                        LexicalizedParser lexicalizedParser,
                                        Model model) {
        this.op = op;
        this.model = model;

        stateIndex = lexicalizedParser.stateIndex;
        wordIndex = lexicalizedParser.wordIndex;
        tagIndex = lexicalizedParser.tagIndex;

        goalStr = lexicalizedParser.treebankLanguagePack().startSymbol();
        bg = lexicalizedParser.bg;
        ug = lexicalizedParser.ug;
        lex = lexicalizedParser.lex;
        numStates = lexicalizedParser.stateIndex.size();

        parallelizer = new Parallelizer(op, (getVocabSize() + 1)/op.trainOp.blockNum);
    }

    public StanfordCompositionalGrammar(Options op,
                                        LexicalizedParser lexicalizedParser) {
        this(op, lexicalizedParser,
                new Model(op.modelOp.dimensions,
                        lexicalizedParser.wordIndex.size(),
                        op.grammarOp.grammarType));
    }

    public void doLexScores(final AbstractInsideOutsideScore score) {
        final StanfordCompositionalInsideOutsideScore s =
                (StanfordCompositionalInsideOutsideScore) score;

        int length = s.getLength();
        Sentence sentence = s.getSentence();
        final int[] words = new int[length];

        for (int i = 0; i < length; i++) {
            String word = sentence.get(i).getSignature();
            words[i] = wordIndex.indexOf(word);
        }

        Function<Integer, Void> lexFunc = new Function<Integer, Void>() {
            @Nullable
            public Void apply(final @Nullable Integer start) {
                final int end = start + 1;
                final int split = start;

                // Set phrase for word sentence[start]
                s.phraseMatrix[start][end] =
                        s.phraseMatrix[start][end]
                                .add(model.word2vec(s.sentence.get(start)));

                s.compositionMatrix[start][end][split] =
                        s.compositionMatrix[start][end][split].add(
                                s.phraseMatrix[start][end]);

                // For leaf nodes, the energy of the node is
                // a function of phrase representation
                final double energy = model.energy(s.phraseMatrix[start][end]);

                log.debug("Doing lex score lookup for index {}", start);

                // For each word, figure out the corresponding tag,
                final int word = words[start];
                Iterator<IntTaggedWord> taggingI;
                for (taggingI = lex.ruleIteratorByWord(word, start, null);
                     taggingI.hasNext(); ) {

                    IntTaggedWord tagging = taggingI.next();
                    int state = stateIndex.indexOf(tagIndex.get(tagging.tag));
                    // score the cell according to P(word|tag) in the lexicon
                    // this is equivalent to log(\theta_r)
                    double lexScore = lex.score(tagging, start,
                            wordIndex.get(tagging.word), null);

                    if (lexScore > Double.NEGATIVE_INFINITY) {

                        // \zeta_{A->w_i}
                        double compScore = exp(-energy + lexScore);

                        // \pi(A, w_i) = \zeta_{A->w_i}
                        s.addToScore(s.iSplitSpanStateScore, compScore, start, end, split, state);
                        s.addToScore(s.iScore, compScore, start, end, state);

                            // \pi (w_i^j) += = \zeta_{A->w_i}
                        synchronized (s.compISplitScore) {
                            s.compISplitScore[start][end][split] += compScore;
                        }

                        synchronized (s.compIScore) {
                            s.compIScore[start][end] += compScore;
                        }
                    }
                }

                Function<Integer, Void> unaryFunc = new Function<Integer, Void>() {
                    @Nullable
                    public Void apply(final @Nullable Integer state) {
                        double iS = s.iScore.getAsDouble(start, end, state);
                        if (iS == 0) {
                            return null;
                        }
                        iS = log(iS);

                        UnaryRule[] unaries = ug.closedRulesByChild(state);
                        for (UnaryRule ur : unaries) {
                            int parentState = ur.parent;
                            double pS = ur.score;

                            // \zeta_{A->w_i}
                            double compScore = exp(iS + pS);

                            // \pi(A, w_i^j) = \zeta_{A->w_i}
                            s.addToScore(s.iSplitSpanStateScore, compScore, start, end, split, parentState);
                            s.addToScore(s.iScore, compScore, start, end, parentState);

                            // \pi (w_i^j) += \zeta_{A->w_i}
                            synchronized (s.compISplitScore) {
                                s.compISplitScore[start][end][split] += compScore;
                            }

                            synchronized (s.compIScore) {
                                s.compIScore[start][end] += compScore;
                            }
                        }
                        return null;
                    }
                };

                if (op.trainOp.parallel) {
                    parallelizer.parallelizer(0, numStates, unaryFunc);
                } else {
                    for (int state = 0; state < numStates; state++) {
                        unaryFunc.apply(state);
                    }
                }
                return null;
            }
        };

        if (op.trainOp.parallel ) {
            parallelizer.parallelizer(0, length, lexFunc);
        } else {
            for (int start = 0; start < length; start++) {
                lexFunc.apply(start);
            }
        }
    } // end doLexScores(List sentence)

    /**
     * Fills in the iScore array of each category over each span
     * of length 2 or more.
     */
    public void doInsideScores(final AbstractInsideOutsideScore score) {
        for (int diff = 2; diff <= score.length; diff++) {
            // usually stop one short because boundary symbol only combines
            // with whole sentence span. So for 3 word sentence + boundary = 4,
            // length == 4, and do [0,2], [1,3]; [0,3]; [0,4]
            for (int start = 0; start < ((diff == score.length) ? 1 : score.length - diff); start++) {
                doInsideChartCell(score, start, start + diff);
            } // for start
        } // for diff (i.e., span)
    } // end doInsideScores()


    /**
     * Compute inside, inside span, inside span split Score for span (start,end).
     *
     * @param start start index of span
     * @param end   end index of span
     */
    private void doInsideChartCell(final AbstractInsideOutsideScore score,
                                   final int start, final int end) {

        final StanfordCompositionalInsideOutsideScore s =
                (StanfordCompositionalInsideOutsideScore)score;

        log.debug("Doing iScore for span {} - {}", start, end);
        final Set<BinaryRule> binaryRuleSet = new HashSet<BinaryRule>();

        // calculate iScore for state by summing over split
        // and iSpanSplitScore by summing over states
        // Stops one short of end as last span should be
        // (start, end - 1), (end - 1, end)
        Function<Integer, Void> iFunc = new Function<Integer, Void>() {
            @Nullable
            public Void apply(@Nullable final Integer split) {

                INDArray child1 = s.phraseMatrix[start][split];
                INDArray child2 = s.phraseMatrix[split][end];

                // Compose parent (start, end) from children
                // (start, split), (split, end)
                synchronized (s.compositionMatrix) {
                    s.compositionMatrix[start][end][split] =
                            s.compositionMatrix[start][end][split].add(
                                    model.compose(child1, child2));
                }

                // Composition energy of parent (start,end)
                // by children (start, split), (split, end)
                final double energy = model.energy(
                        s.compositionMatrix[start][end][split],
                        child1, child2);


                Function<Integer, Void> leftStatFunc = new Function<Integer, Void>() {
                    @Nullable
                    public Void apply(@Nullable Integer leftState) {
                        BinaryRule[] leftRules = bg.splitRulesWithLC(leftState);
                        for (BinaryRule rule : leftRules) {

                            int rightState = rule.rightChild;
                            int parentState = rule.parent;

                            // This binary split might be able to cover the span depending upon children's coverage.
                            double pS = -energy + rule.score;

                            // \pi(B, w_i^k)
                            double lS = s.getScore(s.iScore, start, split, leftState);
                            if (lS == 0f) {
                                continue;
                            }
                            lS = log(lS);

                            // \pi(C, w_{k+1}^j)
                            double rS = s.getScore(s.iScore, split, end, rightState);
                            if (rS == 0f) {
                                continue;
                            }
                            rS = log(rS);

                            synchronized (binaryRuleSet) {
                                binaryRuleSet.add(rule);
                            }

                            // \pi(A,w_i^j -> BC, w_i^k w_{k+1}^j) =
                            //          \zeta_{A,w_i^j -> BC, w_i^k w_{k+1}^j} *
                            //                  \pi(B, w_i^k) * \pi(C, w_{k+1}^j)
                            double compScore = exp(pS + lS + rS);

                            // in left child
                            // \pi(A,w_i^j -> BC, w_i^k w_{k+1}^j)
                            s.addToScore(s.iSplitSpanStateScore, compScore, start, end, split, parentState);
                            // \pi(A,w_i^j) += \pi(A,w_i^j -> BC, w_i^k w_{k+1}^j)
                            s.addToScore(s.iScore, compScore, start, end, parentState);

                            // \pi(w_i^j <- w_i^k w_{k+1}^j) +=  \pi(A,w_i^j -> BC, w_i^k w_{k+1}^j)
                            synchronized (s.compISplitScore) {
                                s.compISplitScore[start][end][split] += compScore;
                            }
                            // pi(w_i^j) += \pi(A,w_i^j -> BC, w_i^k w_{k+1}^j)
                            synchronized (s.compIScore) {
                                s.compIScore[start][end] += compScore;
                            }

                        } // end for leftRules
                        return null;
                    }
                };
                if (op.trainOp.parallel) {
                    parallelizer.parallelizer(0, numStates, leftStatFunc);
                } else {
                    for (int leftState = 0; leftState < numStates; leftState++) {
                        leftStatFunc.apply(leftState);
                    }
                }


                Function<Integer, Void> rightStateFunc = new Function<Integer, Void>() {
                    @Nullable
                    public Void apply(@Nullable Integer rightState) {
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
                            double pS = -energy + rule.score;


                            double lS = s.getScore(s.iScore, start, split, leftState);
                            if (lS == 0f) {
                                continue;
                            }
                            lS = log(lS);

                            double rS = s.getScore(s.iScore, split, end, rightState);
                            if (rS == 0f) {
                                continue;
                            }
                            rS = log(rS);

                            synchronized (binaryRuleSet) {
                                binaryRuleSet.add(rule);
                            }

                            // \pi(A,w_i^j -> BC, w_i^k w_{k+1}^j) =
                            //          \zeta_{A,w_i^j -> BC, w_i^k w_{k+1}^j} *
                            //                  \pi(B, w_i^k) * \pi(C, w_{k+1}^j)
                            double compScore = exp(pS + lS + rS);

                            // right child
                            // \pi(A,w_i^j -> BC, w_i^k w_{k+1}^j)
                            s.addToScore(s.iSplitSpanStateScore, compScore, start, end, split, parentState);
                            // \pi(A,w_i^j) += \pi(A,w_i^j -> BC, w_i^k w_{k+1}^j)
                            s.addToScore(s.iScore, compScore, start, end, parentState);

                            // \pi(w_i^j <- w_i^k w_{k+1}^j) +=  \pi(A,w_i^j -> BC, w_i^k w_{k+1}^j)
                            synchronized (s.compISplitScore) {
                                s.compISplitScore[start][end][split] += compScore;
                            }
                            // pi(w_i^j) += \pi(A,w_i^j -> BC, w_i^k w_{k+1}^j)
                            synchronized (s.compIScore) {
                                s.compIScore[start][end] += compScore;
                            }

                        } // end for rightRules
                        return null;
                    }
                };

                if (op.trainOp.parallel) {
                    parallelizer.parallelizer(0, numStates, rightStateFunc);
                } else {
                    for (int rightState = 0; rightState < numStates; rightState++) {
                        rightStateFunc.apply(rightState);
                    }
                }
                return null;
            }
        };

        if (op.trainOp.parallel) {
            parallelizer.parallelizer(start + 1, end, iFunc);
        } else {
            for (int sp = start + 1; sp < end; sp++) {
                iFunc.apply(sp);
            }
        }


        Function<Integer, Void> unaryFunc = new Function<Integer, Void>() {
            @Nullable
            public Void apply(@Nullable Integer state) {
                double iS = s.getScore(s.iScore, start, end, state);
                if (iS == 0f) {
                    return null;
                }
                iS = log(iS);

                UnaryRule[] unaries = ug.closedRulesByChild(state);
                for (UnaryRule ur : unaries) {

                    int parentState = ur.parent;
                    double pS = ur.score;
                    double tot = exp(iS + pS);

                    s.addToScore(s.iScore, tot, start, end, parentState);
                    synchronized (s.compIScore) {
                        s.compIScore[start][end] += tot;
                    }

                    for (int split = start + 1; split < end; split++) {

                        double iSS = s.getScore(s.iSplitSpanStateScore, start, end, split, state);
                        if (iSS == 0d) {
                            continue;
                        }
                        iSS = log(iSS);
                        double totS = exp(iSS + pS);
                        s.addToScore(s.iSplitSpanStateScore, totS, start, end, split, parentState);

                        synchronized (s.compISplitScore) {
                            s.compISplitScore[start][end][split] += totS;
                        }
                    }
                } // for UnaryRule r
                return null;
            }
        };

        if (op.trainOp.parallel) {
            parallelizer.parallelizer(0, numStates, unaryFunc);
        } else {
            // do unary rules -- one could promote this loop and put start inside
            for (int state = 0; state < numStates; state++) {
                unaryFunc.apply(state);
            } // for unary rules
        }

        for (int split = start + 1; split < end; split++) {
            // X(i,j) * \pi(i,j) = X(i,k,j) * \pi(i,j,k)
            s.phraseMatrix[start][end] =
                    s.phraseMatrix[start][end]
                            .add(s.compositionMatrix[start][end][split]
                                    .mul(s.compISplitScore[start][end][split]));
        }

        // normalize weights to get them to sum to 1.
        // X(i,j) = X(i,k) * \pi(i,j)/\pi(i,j)
        s.phraseMatrix[start][end] =
                s.phraseMatrix[start][end]
                        .div(s.compIScore[start][end]);
    }


    public void doOutsideScores(final AbstractInsideOutsideScore score) {
        final StanfordCompositionalInsideOutsideScore s =
                (StanfordCompositionalInsideOutsideScore) score;

        final int initialStart = 0;
        final int initialEnd = s.length;
        final int startSymbol = stateIndex.indexOf(goalStr);

        s.setScore(s.oScore, 1.0f,
                initialStart, initialEnd, startSymbol);

        for (int diff = s.length; diff >= 1; diff--) {
            for (int st = 0; st + diff <= s.length; st++) {
                final int start = st;
                final int end = st + diff;

                Function<Integer, Void> unaryFunc = new Function<Integer, Void>() {
                    @Nullable
                    public Void apply(@Nullable Integer parentState) {
                        // if current parentState's outside score is zero,
                        // child's would be zero as well
                        double oS = s.getScore(s.oScore, start, end, parentState);
                        if (oS == 0f) {
                            return null;
                        }
                        oS = log(oS);

                        UnaryRule[] rules = ug.closedRulesByParent(parentState);
                        for (UnaryRule ur : rules) {
                            double pS = ur.score;
                            int childState = ur.child;
                            double tot = exp(oS + pS);
                            log.debug("Adding unary rule {} to outside score for Start: {}, End: {}"
                                    , ur, start, end);

                            s.addToScore(s.oScore, tot, start, end, childState);
                        }   // end for unary rule iter
                        return null;
                    }
                };

                log.debug("Doing oScore for span ({}, {})", start, end);

                if (op.trainOp.parallel) {
                    parallelizer.parallelizer(0, numStates, unaryFunc);
                } else {
                    // do unaries
                    for (int parentState = 0; parentState < numStates; parentState++) {
                        unaryFunc.apply(parentState);
                    }   // end for parentState
                }

                Function<Integer, Void> oFunc = new Function<Integer, Void>() {
                    @Nullable
                    public Void apply(@Nullable final Integer split) {
                        INDArray child1 = s.phraseMatrix[start][split];
                        INDArray child2 = s.phraseMatrix[split][end];

                        // Composition energy of parent (start,end)
                        // by children (start, split), (split, end)
                        final double energy = model.energy(
                                s.compositionMatrix[start][end][split],
                                child1, child2);


                        Function<Integer, Void> binaryFunc = new Function<Integer, Void>() {
                            @Nullable
                            public Void apply(@Nullable Integer parentState) {
                                // if current parentState's outside score is zero,
                                // child's would be zero as well
                                double oS = s.getScore(s.oScore, start, end, parentState);
                                if (oS == 0f) {
                                    return null;
                                }
                                oS = log(oS);

                                List<BinaryRule> rules = bg.ruleListByParent(parentState);
                                for (BinaryRule br : rules) {
                                    int leftState = br.leftChild;
                                    int rightState = br.rightChild;

                                    double pS = -energy + br.score;

                                    int lStart = start, lEnd = split;
                                    double rS = s.getScore(s.iScore, split, end, rightState);
                                    if (rS > 0f) {
                                        rS = log(rS);
                                        s.addToScore(s.oScore, exp(pS + rS + oS), lStart, lEnd, leftState);
                                    } // end if rs > 0


                                    // If iScore of the left span is zero, so is the
                                    // oScore of left span
                                    int rStart = split, rEnd = end;
                                    double lS = s.getScore(s.iScore, start, split, leftState);
                                    if (lS > 0f) {
                                        lS = log(lS);
                                        s.addToScore(s.oScore, exp(pS + lS + oS), rStart, rEnd, rightState);
                                    }   // end if ls > 0
                                }
                                return null;
                            }
                        };

                        if (op.trainOp.parallel) {
                            parallelizer.parallelizer(0, numStates, binaryFunc);
                        } else {
                            // do binaries
                            for (int parentState = 0; parentState < numStates; parentState++) {
                                binaryFunc.apply(parentState);
                            }   // end for parent state
                        }
                        return null;
                    }
                };

                if (op.trainOp.parallel) {
                    parallelizer.parallelizer(start + 1, end, oFunc);
                } else {
                    for (int split = start + 1; split < end; split++) {
                        oFunc.apply(split);
                    }   // end for split
                }
            }   // end for start
        }   // end for end
    }   // end doOutsideScores

    public void doOutsideScores2(final AbstractInsideOutsideScore score) {
        final StanfordCompositionalInsideOutsideScore s =
                (StanfordCompositionalInsideOutsideScore) score;

        final int initialStart = 0;
        final int initialEnd = s.length;
        final int startSymbol = stateIndex.indexOf(goalStr);

        s.setScore(s.oScore, 1.0f,
                initialStart, initialEnd, startSymbol);

        for (int diff = s.length; diff >= 1; diff--) {
            for (int st = 0; st + diff <= s.length; st++) {
                final int start = st;
                final int end = st + diff;

                Function<Integer, Void> unaryFunc = new Function<Integer, Void>() {
                    @Nullable
                    public Void apply(@Nullable Integer parentState) {
                        // if current parentState's outside score is zero,
                        // child's would be zero as well
                        double oS = s.getScore(s.oScore, start, end, parentState);
                        if (oS == 0f) {
                            return null;
                        }
                        oS = log(oS);

                        UnaryRule[] rules = ug.closedRulesByParent(parentState);
                        for (UnaryRule ur : rules) {
                            double pS = ur.score;
                            int childState = ur.child;
                            double tot = exp(oS + pS);
                            log.debug("Adding unary rule {} to outside score for Start: {}, End: {}"
                                    , ur, start, end);

                            s.addToScore(s.oScore, tot, start, end, childState);
                        }   // end for unary rule iter
                        return null;
                    }
                };

                log.debug("Doing oScore for span ({}, {})", start, end);

                if (op.trainOp.parallel) {
                    parallelizer.parallelizer(0, numStates, unaryFunc);
                } else {
                    // do unaries
                    for (int parentState = 0; parentState < numStates; parentState++) {
                        unaryFunc.apply(parentState);
                    }   // end for parentState
                }

                Function<Integer, Void> oFunc = new Function<Integer, Void>() {
                    @Nullable
                    public Void apply(@Nullable final Integer split) {
                        INDArray child1 = s.phraseMatrix[start][split];
                        INDArray child2 = s.phraseMatrix[split][end];

                        // Composition energy of parent (start,end)
                        // by children (start, split), (split, end)
                        final double energy = model.energy(
                                s.compositionMatrix[start][end][split],
                                child1, child2);


                        Function<Integer, Void> binaryFuncLeft = new Function<Integer, Void>() {
                            @Nullable
                            public Void apply(@Nullable Integer leftState) {
                                BinaryRule[] rules = bg.splitRulesWithLC(leftState);
                                for (BinaryRule br : rules) {
                                    int parentState = br.parent;
                                    int rightState = br.rightChild;

                                    // if current parentState's outside score is zero,
                                    // child's would be zero as well
                                    double oS = s.getScore(s.oScore, start, end, parentState);
                                    if (oS == 0f) {
                                        continue;
                                    }
                                    oS = log(oS);

                                    double pS = -energy + br.score;

                                    int lStart = start, lEnd = split;
                                    double rS = s.getScore(s.iScore, split, end, rightState);
                                    if (rS > 0f) {
                                        rS = log(rS);
                                        s.addToScore(s.oScore, exp(pS + rS + oS), lStart, lEnd, leftState);
                                    } // end if rs > 0
                                }
                                return null;
                            }
                        };

                        if (op.trainOp.parallel) {
                            parallelizer.parallelizer(0, numStates, binaryFuncLeft);
                        } else {
                            // do binaries
                            for (int leftState = 0; leftState < numStates; leftState++) {
                                binaryFuncLeft.apply(leftState);
                            }   // end for parent state
                        }

                        Function<Integer, Void> binaryFuncRight = new Function<Integer, Void>() {
                            @Nullable
                            public Void apply(@Nullable Integer rightState) {
                                // if current parentState's outside score is zero,
                                // child's would be zero as well


                                BinaryRule[] rules = bg.splitRulesWithRC(rightState);
                                for (BinaryRule br : rules) {
                                    int leftState = br.leftChild;
                                    int parentState = br.parent;

                                    double pS = -energy + br.score;

                                    double oS = s.getScore(s.oScore, start, end, parentState);
                                    if (oS == 0f) {
                                        continue;
                                    }
                                    oS = log(oS);

                                    // If iScore of the left span is zero, so is the
                                    // oScore of left span
                                    int rStart = split, rEnd = end;
                                    double lS = s.getScore(s.iScore, start, split, leftState);
                                    if (lS > 0f) {
                                        lS = log(lS);
                                        s.addToScore(s.oScore, exp(pS + lS + oS), rStart, rEnd, rightState);
                                    }   // end if ls > 0
                                }
                                return null;
                            }
                        };

                        if (op.trainOp.parallel) {
                            parallelizer.parallelizer(0, numStates, binaryFuncRight);
                        } else {
                            // do binaries
                            for (int rightState = 0; rightState < numStates; rightState++) {
                                binaryFuncRight.apply(rightState);
                            }   // end for parent state
                        }
                        return null;
                    }
                };

                if (op.trainOp.parallel) {
                    parallelizer.parallelizer(start + 1, end, oFunc);
                } else {
                    for (int split = start + 1; split < end; split++) {
                        oFunc.apply(split);
                    }   // end for split
                }
            }   // end for start
        }   // end for end
    }   // end doOutsideScores


    /**
     * Populate mu score arrays
     */
    public void doMuScore(final AbstractInsideOutsideScore score) {
        final StanfordCompositionalInsideOutsideScore s =
                (StanfordCompositionalInsideOutsideScore) score;

        // Handle lead node case.
        // There is no split here and span value
        // is stored at start
        for (int st = 0; st < s.length; st++) {
            final int start = st;
            final int end = st + 1;
            final int split = st;
            log.debug("Doing muScore for span {} - {}", start, end);

            Function<Integer, Void> unaryFunc = new Function<Integer, Void>() {
                @Nullable
                public Void apply(@Nullable Integer state) {
                    // If iScore or oScore is zero, the mu score is zero
                    double iS = s.getScore(s.iSplitSpanStateScore, start, end, split, state);
                    if (iS == 0) {
                        return null;
                    }
                    iS = log(iS);

                    double oS = s.getScore(s.oScore, start, end, state);
                    if (oS == 0f) {
                        return null;
                    }
                    oS = log(oS);

                    synchronized (s.compositionalMu) {
                        s.compositionalMu[start][end][split] += exp(iS + oS);
                    }

                    return null;
                }
            };

            if (op.trainOp.parallel) {
                parallelizer.parallelizer(0, numStates, unaryFunc);
            } else {
                for (int state = 0; state < numStates; state++)
                {
                    unaryFunc.apply(state);
                }
            }
        }

        // Handle cases for diff = 2 and above
        for (int df = 2; df <= s.length; df++) {
            final int diff = df;
            for (int st = 0; st + df <= s.length; st++) {
                final int start = st;
                final int end = st + diff;

                log.debug("Doing muScore for span {} - {}", start, end);
                for (int sp = start + 1; sp < end; sp++) {
                    final int split = sp;
                    Function<Integer, Void> binaryFunc = new Function<Integer, Void>() {
                        @Nullable
                        public Void apply(@Nullable Integer state) {

                            double iS = s.getScore(s.iSplitSpanStateScore, start, end, split, state);
                            if (iS == 0) {
                                return null;
                            }
                            iS = log(iS);

                            // If iScore or oScore is zero, the mu score is zero
                            double oS = s.getScore(s.oScore, start, end, state);
                            if (oS == 0) {
                                return null;
                            }
                            oS = log(oS);

                            synchronized (s.compositionalMu) {
                                s.compositionalMu[start][end][split] += exp(oS + iS);
                            }
                            return null;
                        }
                    };

                    if (op.trainOp.parallel) {
                        parallelizer.parallelizer(0, numStates, binaryFunc);
                    } else {
                        for (int state = 0; state < numStates; state++) {
                            binaryFunc.apply(state);
                        }
                    }
                }
            }   // end for start
        }   // end for diff
    }


    /**
     * Compute inside and outside score for the sentence.
     * Also computes span and span split score we need.
     */
    public void computeInsideOutsideProb(AbstractInsideOutsideScore score) {
        StanfordCompositionalInsideOutsideScore s =
                (StanfordCompositionalInsideOutsideScore) score;

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

  //      s.postProcess();

        log.info("Compositional Score for sentence#{}:: {} => {}",
                idx, sz, s.getSentenceScore());

        if (op.debug) {
            log.info("Memory Size StanfordIOScore: {}:: {}\n" +
                            "\t {} => {} MB\n" +
                            "\t {} => {} MB\n" +
                            "\t {} => {} MB\n" +
                            "\t {} => {} MB\n" +
                            "\t {} => {} MB\n" +
                            "total => {} MB",
                    idx, sz,
                    "compIScore", getSize(s.compIScore),
                    "compISplitScore", getSize(s.compISplitScore),
                    "compositionalMu", getSize(s.compositionalMu),
                    "compositionMatrix", getSize(s.compositionMatrix),
                    "phraseMatrix", getSize(s.phraseMatrix),
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
        StanfordCompositionalInsideOutsideScore score =
                new StanfordCompositionalInsideOutsideScore(
                        sentence, op.modelOp.dimensions, numStates);
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
            signature = str.toLowerCase();
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