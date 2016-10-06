package com.kushalarora.compositionalLM.lang;

import com.google.common.base.Function;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.options.Options;
import com.kushalarora.compositionalLM.utils.Parallelizer;
import edu.stanford.nlp.parser.lexparser.BinaryGrammar;
import edu.stanford.nlp.parser.lexparser.BinaryRule;
import edu.stanford.nlp.parser.lexparser.IntTaggedWord;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.parser.lexparser.Lexicon;
import edu.stanford.nlp.parser.lexparser.UnaryGrammar;
import edu.stanford.nlp.parser.lexparser.UnaryRule;
import edu.stanford.nlp.util.Index;
import java.util.Iterator;
import java.util.List;
import javax.annotation.Nullable;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang.StringUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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

    protected final int blockSize;

    public StanfordCompositionalGrammar(Options op,
                                        LexicalizedParser lexicalizedParser,
                                        Model model,
                                        Parallelizer parallelizer) {
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
        blockSize = (getVocabSize() + 1)/op.trainOp.blockNum;
        this.parallelizer = parallelizer;
    }

    public StanfordCompositionalGrammar(Options op,
                                        LexicalizedParser lexicalizedParser,
                                        Parallelizer parallelizer) {
        this(op, lexicalizedParser,
                new Model(op, op.modelOp.dimensions,
                        lexicalizedParser.wordIndex.size(),
                        op.grammarOp.grammarType), parallelizer);
    }

    public void doLexScores(final AbstractInsideOutsideScore score) {
        final StanfordCompositionalInsideOutsideScore s =
                (StanfordCompositionalInsideOutsideScore) score;

        final int length = s.getLength();
        final Sentence sentence = s.getSentence();
        final int[] words = new int[length];

        for (int i = 0; i < length; i++) {
            String word = sentence.get(i).getSignature();
            words[i] = wordIndex.indexOf(word);
        }

        for (int st = 0; st < length; st++) {
            final int start = st;
            final int end = start + 1;
            final int split = start;


	        // Set phrase for word sentence[start]
	        s.phraseMatrix[start][end] =
		        s.phraseMatrix[start][end]
			        .add(model.word2vec(s.sentence.get(start)));

	        s.compositionMatrix[start][end][split] =
		        s.compositionMatrix[start][end][split]
			        .add(s.phraseMatrix[start][end]);

	        // For leaf nodes, the energy of the node is
	        // a function of phrase representation
	        final double pWord = model.probabilityWord(s.phraseMatrix[start][end]);

	        log.debug("Doing lex score lookup for index {}", start);

	        // For each word, figure out the corresponding tag,
	        final int word = words[start];
	        Iterator<IntTaggedWord> taggingI;

	        for (taggingI = lex.ruleIteratorByWord(word, start, null); taggingI.hasNext(); ) {

		        IntTaggedWord tagging = taggingI.next();
		        int state = stateIndex.indexOf(tagIndex.get(tagging.tag));
		        // score the cell according to P(word|tag) in the lexicon
		        // this is equivalent to log(\theta_r)
		        double lexScore = lex.score(tagging,
                                            start,
			                                wordIndex.get(tagging.word),
                                            null);

		        if (lexScore > Double.NEGATIVE_INFINITY) {

			        // \pi(A->w_i)
                    final double theta_w_i = exp(lexScore);
			        s.addToScore(s.iScorePCFG, theta_w_i, start, end, state);

                    s.compIScorePCFG[start][end] += theta_w_i;

			        // \zeta_{A->w_i}
			        final double zeta_w_i = exp(log(pWord) + lexScore);

                    // \pi (w_i^j) += = \zeta_{A->w_i}
                    s.addToScore(s.iScore, zeta_w_i, start, end, state);

                    // \pi(A, w_i) = \zeta_{A->w_i}
			        s.addToScore(s.iSplitSpanStateScore, zeta_w_i, start, end, split, state);

			        s.compISplitScore[start][end][split] += zeta_w_i;

			        s.compIScore[start][end] += zeta_w_i;

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

			        double iS_pcfg = s.iScorePCFG.getAsDouble(start, end, state);
			        if (iS_pcfg == 0) {
				        return null;
			        }

                    iS_pcfg = log(iS_pcfg);

			        double cumlCompScore = 0f;

			        UnaryRule[] unaries = ug.closedRulesByChild(state);
			        for (UnaryRule ur : unaries) {
				        int parentState = ur.parent;
				        double pS = ur.score;

				        // \pi(A->w_i)
                        final double theta_w_i = exp(iS_pcfg + pS);
				        s.addToScore(s.iScorePCFG, theta_w_i, start, end, parentState);

                        s.compIScorePCFG[start][end] += theta_w_i;

				        // \zeta_{A->w_i^j}
				        double zeta_A_w_i_j = exp(iS + pS);

				        // \pi (w_i^j) += \zeta_{A->w_i}
				        s.addToScore(s.iScore, zeta_A_w_i_j, start, end, parentState);

				        // \pi(A, w_i^j) = \zeta_{A->w_i}
				        s.addToScore(s.iSplitSpanStateScore, zeta_A_w_i_j, start, end, split, parentState);

				        cumlCompScore += zeta_A_w_i_j;
			        }

			        if (cumlCompScore > 0) {

				        synchronized (s.compISplitScore) {
					        s.compISplitScore[start][end][split] += cumlCompScore;
				        }

				        synchronized (s.compIScore) {
					        s.compIScore[start][end] += cumlCompScore;
				        }
			        }

			        return null;
		        }
	        };

	        if (op.trainOp.modelParallel) {
		        parallelizer.parallelizer(0, numStates, unaryFunc, blockSize);
	        } else {
		        for (int state = 0; state < numStates; state++) {
			        unaryFunc.apply(state);
		        }
	        }
        }
    } // end doLexScores(List sentence)

    /**
     * Fills in the iScore array of each category over each span
     * of length 2 or more.
     */
    public void doInsideScores(final AbstractInsideOutsideScore score) {
        final StanfordCompositionalInsideOutsideScore s =
                (StanfordCompositionalInsideOutsideScore) score;
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
    private void doInsideChartCell(final AbstractInsideOutsideScore score,
                                   final int start, final int end) {

        final StanfordCompositionalInsideOutsideScore s =
                (StanfordCompositionalInsideOutsideScore) score;

        s.binaryRuleSet.clear();

        log.debug("Doing iScore for span {} - {}", start, end);

        // calculate iScore for state by summing over split
        // and iSpanSplitScore by summing over states
        // Stops one short of end as last span should be
        // (start, end - 1), (end - 1, end)

        final double[] pComp = new double[end];
        double Z_energy_split = 0;
        for (int sp = start + 1; sp < end; sp++) {
            final INDArray child1 = s.phraseMatrix[start][sp];
            final INDArray child2 = s.phraseMatrix[sp][end];

            // Compose parent (start, end) from children
            // (start, split), (split, end)
            s.compositionMatrix[start][end][sp] =
                    s.compositionMatrix[start][end][sp]
                        .addi(model.compose(child1, child2));

            // Composition energy of parent (start,end)
            // by children (start, split), (split, end)
            double unProbComp = model.unProbabilityComp(s.compositionMatrix[start][end][sp], child1, child2);

            pComp[sp] = unProbComp;

            Z_energy_split += unProbComp;
        }

        for (int sp = start + 1; sp < end; sp++) {
            pComp[sp] = pComp[sp]/Z_energy_split;
        }

        for (int sp = start + 1; sp < end; sp++) {
            final int split = sp;
            Function<Integer, Void> leftStatFunc = new Function<Integer, Void>() {
                @Nullable
                public Void apply(@Nullable Integer leftState) {
                    BinaryRule[] leftRules = bg.splitRulesWithLC(leftState);

                    double cumlCompScore = 0;
                    for (BinaryRule rule : leftRules) {

                        int rightState = rule.rightChild;
                        int parentState = rule.parent;

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

                        double ls_pcfg = s.getScore(s.iScorePCFG, start, split, leftState);
                        if (ls_pcfg == 0f) {
                            continue;
                        }
                        ls_pcfg = log(ls_pcfg);

                        double rS_pcfg = s.getScore(s.iScorePCFG, split, end, rightState);
                        if (rS_pcfg == 0f) {
                            continue;
                        }
                        rS_pcfg = log(rS_pcfg);

                        // Its a synchronized set
                        // so should allow one entry at a
                        // time.
                        s.binaryRuleSet.add(rule);

                        // pi(A, w_i^j, BC, w_i^k, w_i_{k+1}^j)
                        final double theta_A_BC = exp(rule.score + ls_pcfg + rS_pcfg);
	                    s.addToScore(s.iScorePCFG, theta_A_BC, start, end, parentState);

                        s.compIScorePCFG[start][end] += theta_A_BC;

	                    // This binary split might be able to cover the span depending upon children's coverage.
                        double pS = log(pComp[split]) + rule.score;

                        // \zeta_(A,w_i^j -> BC, w_i^k w_{k+1}^j) =
                        //          \pi{A,w_i^j -> BC, w_i^k w_{k+1}^j} *
                        //                  \pi(B, w_i^k) * \pi(C, w_{k+1}^j)
                        double compScore = exp(pS + lS + rS);

                        // in left child
                        // \pi(A,w_i^j -> BC, w_i^k w_{k+1}^j)
                        s.addToScore(s.iSplitSpanStateScore,
	                                    compScore, start, end,
	                                    split, parentState);

                        // \pi(A,w_i^j) += \pi(A,w_i^j -> BC, w_i^k w_{k+1}^j)
                        s.addToScore(s.iScore, compScore, start, end, parentState);

                        cumlCompScore += compScore;

                    } // end for leftRules

                    if (cumlCompScore > 0) {

                        // \pi(w_i^j <- w_i^k w_{k+1}^j) +=  \pi(A,w_i^j -> BC, w_i^k w_{k+1}^j)
                        synchronized (s.compISplitScore) {
                            s.compISplitScore[start][end][split] += cumlCompScore;
                        }

                        // pi(w_i^j) += \pi(A,w_i^j -> BC, w_i^k w_{k+1}^j)
                        synchronized (s.compIScore) {
                            s.compIScore[start][end] += cumlCompScore;
                        }
                    }
                    return null;
                }
            };
            if (op.trainOp.modelParallel) {
                parallelizer.parallelizer(0, numStates, leftStatFunc, blockSize);
            } else {
                for (int leftState = 0; leftState < numStates; leftState++) {
                    leftStatFunc.apply(leftState);
                }
            }


            Function<Integer, Void> rightStateFunc = new Function<Integer, Void>() {
                @Nullable
                public Void apply(@Nullable Integer rightState) {
                    BinaryRule[] rightRules = bg.splitRulesWithRC(rightState);

                    double cumlCompScore = 0;
                    for (BinaryRule rule : rightRules) {

                        // Rule already processed by left state loop
                        if (s.binaryRuleSet.contains(rule)) {
                            log.debug("Rule {} already processed by left child loop.Skipping", rule);
                            continue;
                        }

                        int leftState = rule.leftChild;
                        int parentState = rule.parent;

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


                        double lS_pcfg = s.getScore(s.iScorePCFG, start, split, leftState);
                        if (lS_pcfg == 0f) {
                            continue;
                        }
                        lS_pcfg = log(lS_pcfg);

                        double rS_pcfg = s.getScore(s.iScorePCFG, split, end, rightState);
                        if (rS_pcfg == 0f) {
                            continue;
                        }
                        rS_pcfg = log(rS_pcfg);

                        s.binaryRuleSet.add(rule);

                        final double theta_A_BC = exp(rule.score + lS_pcfg + rS_pcfg);
	                    s.addToScore(s.iScorePCFG, theta_A_BC,
		                                start, end, parentState);

                        s.compIScorePCFG[start][end] += theta_A_BC;

	                    // This binary split might be able to cover the span depending upon children's coverage.
                        double pS = log(pComp[split]) + rule.score;

                        // \pi(A,w_i^j -> BC, w_i^k w_{k+1}^j) =
                        //          \zeta_{A,w_i^j -> BC, w_i^k w_{k+1}^j} *
                        //                  \pi(B, w_i^k) * \pi(C, w_{k+1}^j)
                        double compScore = exp(pS + lS + rS);

                        // right child
                        // \pi(A,w_i^j -> BC, w_i^k w_{k+1}^j)
                        s.addToScore(s.iSplitSpanStateScore, compScore, start, end, split, parentState);

                        // \pi(A,w_i^j) += \pi(A,w_i^j -> BC, w_i^k w_{k+1}^j)
                        s.addToScore(s.iScore, compScore, start, end, parentState);

                        cumlCompScore += compScore;

                    } // end for rightRules

                    if (cumlCompScore > 0) {
                        // \pi(w_i^j <- w_i^k w_{k+1}^j) +=  \pi(A,w_i^j -> BC, w_i^k w_{k+1}^j)
                        synchronized (s.compISplitScore) {
                            s.compISplitScore[start][end][split] += cumlCompScore;
                        }

                        // pi(w_i^j) += \pi(A,w_i^j -> BC, w_i^k w_{k+1}^j)
                        synchronized (s.compIScore) {
                            s.compIScore[start][end] += cumlCompScore;
                        }
                    }
                    return null;
                }
            };


            if (op.trainOp.modelParallel) {
                parallelizer.parallelizer(0, numStates, rightStateFunc, blockSize);
            } else {
                for (int rightState = 0; rightState < numStates; rightState++) {
                    rightStateFunc.apply(rightState);
                }
            }
        }

        for (int sp = start + 1; sp < end; sp++) {
            final int split = sp;
            Function<Integer, Void> unaryFuncSplit = new Function<Integer, Void>() {
                @Nullable
                public Void apply(@Nullable Integer state) {
                    double iSS = s.getScore(s.iSplitSpanStateScore,
                                            start, end, split, state);
                    if (iSS == 0d) {
                        return null;
                    }

                    iSS = log(iSS);
                    UnaryRule[] unaries = ug.closedRulesByChild(state);
                    double cumlUnaryScore = 0;
                    for (UnaryRule ur : unaries) {
                        int parentState = ur.parent;
                        double pS = ur.score;
                        double unaryScore = exp(iSS + pS);
                        s.addToScore(s.iSplitSpanStateScore,
                            unaryScore, start, end, split, parentState);

                        cumlUnaryScore += unaryScore;
                    }

                    if (cumlUnaryScore > 0) {
                        synchronized (s.compISplitScore) {
                            s.compISplitScore[start][end][split] += cumlUnaryScore;
                        }
                    }
                    return null;
                }
            };

            if (op.trainOp.modelParallel) {
                parallelizer.parallelizer(0, numStates, unaryFuncSplit, blockSize);
            } else {
                // do unary rules -- one could promote this loop and put start inside
                for (int state = 0; state < numStates; state++) {
                    unaryFuncSplit.apply(state);
                } // for unary rules
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

                double iS_pcfg = s.getScore(s.iScorePCFG, start, end, state);
                if (iS_pcfg == 0f) {
                    return null;
                }
                iS_pcfg = log(iS_pcfg);

                double cumlUnaryScore = 0;
                UnaryRule[] unaries = ug.closedRulesByChild(state);
                for (UnaryRule ur : unaries) {
                    int parentState = ur.parent;
                    double pS = ur.score;

                    s.addToScore(s.iScorePCFG,
                        exp(iS_pcfg + pS),
                        start, end, parentState);

                    s.compIScorePCFG[start][end] += exp(iS_pcfg + pS);

                    double unaryScore = exp(iS + pS);
                    s.addToScore(s.iScore, unaryScore,
                        start, end, parentState);

                    cumlUnaryScore += unaryScore;

                } // for UnaryRule r

                if (cumlUnaryScore > 0) {
                    synchronized (s.compIScore) {
                        s.compIScore[start][end] += cumlUnaryScore;
                    }
                }
                return null;
            }
        };

        if (op.trainOp.modelParallel) {
            parallelizer.parallelizer(0, numStates, unaryFunc, blockSize);
        } else {
            // do unary rules -- one could promote this loop and put start inside
            for (int state = 0; state < numStates; state++) {
                unaryFunc.apply(state);
            } // for unary rules
        }


        for (int split = start + 1; split < end; split++) {

            if (s.compIScore[start][end] == 0) {
                continue;
            }

            final double splitNorm =  s.compISplitScore[start][end][split]/s.compIScore[start][end];
            // X(i,j) * \pi(i,j) = X(i,k,j) * \pi(i,j,k)
            s.phraseMatrix[start][end] =
                s.phraseMatrix[start][end]
                .addi(s.compositionMatrix[start][end][split]
                        .mul(splitNorm));
        }

    }


    public void doOutsideScores(final AbstractInsideOutsideScore score) {
        final StanfordCompositionalInsideOutsideScore s =
                (StanfordCompositionalInsideOutsideScore) score;

        final int initialStart = 0;
        final int initialEnd = s.length;
        final int startSymbol = stateIndex.indexOf(goalStr);

        // TODO: Figure out if this makes sense or not.
        int iSTotal = 0;
        for (int state = 0; state < numStates; state++) {
            double iS = s.getScore(s.iScore, initialStart, initialEnd, state);
            if (iS == 0) {
                continue;
            }
            iSTotal += 1;
        }

        for (int state = 0; state < numStates; state++) {
            double iS = s.getScore(s.iScore, initialStart, initialEnd, state);
            if (iS == 0) {
                continue;
            }
            s.addToScore(s.oScore, 1.0/iSTotal, initialStart, initialEnd, state);

        }

        for (int diff = s.length; diff >= 1; diff--) {
            for (int st = 0; st + diff <= s.length; st++) {
                final int start = st;
                final int end = st + diff;
                log.debug("Doing oScore for span ({}, {})", start, end);

               /* Function<Integer, Void> unaryFunc = new Function<Integer, Void>() {
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



                if (op.trainOp.modelParallel) {
                   // parallelizer.parallelizer(0, numStates, unaryFunc, blockSize);
                } else {
                    // do unaries
                    for (int parentState = 0; parentState < numStates; parentState++) {
                        unaryFunc.apply(parentState);
                    }   // end for parentState
                }*/

                final double[] pCompArr = new double[end];
                double ZEnergySplit = 0;
                for (int sp = start + 1; sp < end; sp++) {
                    final int split = sp;
                    INDArray child1 = s.phraseMatrix[start][split];
                    INDArray child2 = s.phraseMatrix[split][end];

                    // Composition energy of parent (start,end)
                    // by children (start, split), (split, end)
                    final double unProbComp =
                        model.unProbabilityComp(
                            s.compositionMatrix[start][end][split],
                            child1, child2);


                    pCompArr[sp] = unProbComp;

                    ZEnergySplit += unProbComp;
                }

                for (int sp = start + 1; sp < end; sp++) {
                    pCompArr[sp] = pCompArr[sp]/ZEnergySplit;
                }

                for (int sp = start + 1; sp < end; sp++) {
                    final int split = sp;
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

                                double pS = log(pCompArr[split]) + br.score;

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

                    if (op.trainOp.modelParallel) {
                        parallelizer.parallelizer(0, numStates, binaryFunc, blockSize);
                    } else {
                        // do binaries
                        for (int parentState = 0; parentState < numStates; parentState++) {
                            binaryFunc.apply(parentState);
                        }   // end for parent state
                    }
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

        // Handle leaf node case.
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

            if (op.trainOp.modelParallel) {
                parallelizer.parallelizer(0, numStates, unaryFunc, blockSize);
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

                    if (op.trainOp.modelParallel) {
                        parallelizer.parallelizer(0, numStates, binaryFunc, blockSize);
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
    public void computeInsideOutsideProb(final AbstractInsideOutsideScore score) {
        final StanfordCompositionalInsideOutsideScore s =
                (StanfordCompositionalInsideOutsideScore) score;

        int idx = s.sentence.getIndex();
        int sz = s.sentence.size();

        log.info("Starting inside score computation:{}::{}", idx, sz);
        long insideTimeStart = System.currentTimeMillis();
        doLexScores(s);

        doInsideScores(s);

        s.setZ_theta_r_w(s.compIScorePCFG[0][s.getLength()]);


        long insideTimeEnd = System.currentTimeMillis();
        log.info("Computed inside score computation:{}::{} in {} secs", idx, sz, (insideTimeEnd - insideTimeStart));

        log.info("Start outside score computation:{}::{}", idx, sz);
        long outsideTimeStart = System.currentTimeMillis();
        doOutsideScores(s);
        long outsideTimeEnd = System.currentTimeMillis();
        log.info("Computed outside score computation:{}::{} in {} secs", idx, sz, (outsideTimeEnd - outsideTimeStart));


        log.info("Start mu score computation:{}::{}", idx, sz);
        long muTimeStart = System.currentTimeMillis();
        doMuScore(s);
        long muTimeEnd = System.currentTimeMillis();
        log.info("Computed mu score computation:{}::{} in {} secs", idx, sz, (muTimeEnd - muTimeStart));

        s.postProcess();

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

    public AbstractInsideOutsideScore getInsideScore(final Sentence sentence, final boolean addEOS) {
        final StanfordCompositionalInsideOutsideScore s =
                new StanfordCompositionalInsideOutsideScore(
                        sentence, op.modelOp.dimensions, numStates, addEOS);

        int idx = s.sentence.getIndex();
        int sz = s.sentence.size();
        log.info("Starting inside score computation:{}::{}", idx, sz);
        doLexScores(s);
        doInsideScores(s);

        s.setZ_theta_r_w(s.compIScorePCFG[0][s.getLength()]);

        log.info("Computed inside score computation:{}::{}", idx, sz);
        return s;
    }

    public double getQScore(StanfordCompositionalInsideOutsideScore score) {
        int length = score.length;
        double p_W = score.compIScore[0][length];
        double qScore = 0;
        INDArray[][] phraseMatrix = new INDArray[length][length + 1];

        Sentence sentence = score.getSentence();
        for (int start = 0; start < length; start++) {
            int end = start + 1;
            phraseMatrix[start][end] = Nd4j.zeros(model.getDimensions(), 1);
            phraseMatrix[start][end] = phraseMatrix[start][end]
                                        .addi(model.word2vec(sentence.get(start)));

            qScore += log(model.probabilityWord(phraseMatrix[start][end]))
	                        * score.compositionalMu[start][end][start]/ p_W;
            if (!Double.isFinite(qScore)) {
                log.info("qScore infinite");
            }
        }

        for (int diff = 2; diff <= length; diff++) {
            for (int start = 0; start + diff <= length  ; start++) {
                int end = start + diff;
                phraseMatrix[start][end] = Nd4j.zeros(model.getDimensions(), 1);
	            double zUnProbComp = 0;
	            double[] unProbComp = new double[length];

                if (score.compIScore[start][end] == 0) {
                    continue;
                }

                for (int split = start + 1; split < end; split++) {
                    INDArray child1 = phraseMatrix[start][split];
                    INDArray child2 = phraseMatrix[split][end];
                    INDArray compVector = model.compose(child1, child2);

                    unProbComp[split] = model.unProbabilityComp(compVector, child1, child2);
                    zUnProbComp += unProbComp[split];


                    final double splitNorm =
                        score.compISplitScore[start][end][split]/score.compIScore[start][end];

                    phraseMatrix[start][end] =
                        phraseMatrix[start][end]
		                .addi(compVector
		                        .muli(splitNorm));

                }

	            for (int split = start + 1; split < end; split++) {

		            qScore += log(unProbComp[split]/zUnProbComp)
                                * score.compositionalMu[start][end][split]/p_W;
                    if (!Double.isFinite(qScore)) {
                        log.info("qScore infinite");
                    }
	            }

            }
        }
        return -1.0 * qScore;
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
        final StanfordCompositionalInsideOutsideScore score =
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
    // TODO: Remove this vocab thing.
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