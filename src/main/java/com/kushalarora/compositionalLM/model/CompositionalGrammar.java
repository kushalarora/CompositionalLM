package com.kushalarora.compositionalLM.model;

import static com.kushalarora.compositionalLM.utils.ObjectSizeFetcher.getSize;
import static java.lang.Math.exp;

import java.io.Serializable;
import java.util.List;

import javax.annotation.Nullable;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.ujmp.core.SparseMatrix;

import com.google.common.base.Function;
import com.kushalarora.compositionalLM.lang.IInsideOutsideScore;
import com.kushalarora.compositionalLM.lang.Sentence;
import com.kushalarora.compositionalLM.lang.Word;
import com.kushalarora.compositionalLM.options.Options;
import com.kushalarora.compositionalLM.utils.Parallelizer;

import lombok.Getter;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import lombok.val;

/**
 * Created by karora on 6/21/15.
 */
@Slf4j
@Getter
public class CompositionalGrammar implements Serializable {
    private final Options op;
    private Model model;
    private Parallelizer parallelizer;

    public CompositionalGrammar(final Model model, final Options op) {
        this.model = model;
        this.op = op;
        parallelizer = new Parallelizer(op, op.grammarOp.maxLength/op.trainOp.blockNum + 1);
    }

    public void parse(List<Word> sentence) {

    }

    public void nbest(List<List<Word>> decodings) {

    }

    public CompositionalInsideOutsideScore getScore(
            Sentence sentence, IInsideOutsideScore preScores) {
        CompositionalInsideOutsideScore score = new CompositionalInsideOutsideScore(sentence, model.getDimensions());
        computeCompInsideOutsideScores(score, preScores);
        return score;
    }


    /**
     * Calculate compositional iScore, composition scores,
     * cumulative composition score and phrase and composition matrix.
     */
    public void doInsideScore(final CompositionalInsideOutsideScore s, final IInsideOutsideScore preScores) {

        final SparseMatrix iScores = preScores.getInsideSpanProb();
        final SparseMatrix iSplitScores = preScores.getInsideSpanSplitProb();

        Function<Integer, Void> insideFuncUnary =
                new Function<Integer, Void>()
                {
                    @Nullable
                    public Void apply(Integer start)
                    {
                        int end = start + 1;
                        int split = start;
                        log.debug("Computing Compositional inside Score for span ({}, {})", start, end);

                        // Set phrase for word sentence[start]
                        s.phraseMatrix[start][end] = s.phraseMatrix[start][end].add(
                                model.word2vec(s.sentence.get(start)));

                        s.compositionMatrix[start][end][split] =
                                s.compositionMatrix[start][end][split].add(
                                        s.phraseMatrix[start][end]);

                        // For leaf nodes, the energy of the node is
                        // a function of phrase representation
                        val energy = model.energy(s.phraseMatrix[start][end]);

                        // un-normalized prob or score, is negative exp of energy
                        double score = ((double) exp(-energy));

                        s.cumlCompositionScore[start][end] += score;
                        s.compositionScore[start][end][split] += score;

                        // comp iScore for a leaf node is iScore of span (start, start + 1)
                        // into composition score of (start, start + 1)
                        // This is like re-writing unary rule a=>w_i
                        double iScore_start_end = preScores.getScore(iScores, start, end);
                        s.compositionalIScore[start][end] += score * iScore_start_end;
                        s.compositionISplitScore[start][end][split] += score * iScore_start_end;
                        return null;
                    }
                };

        if (op.trainOp.parallel) {
            parallelizer.parallelizer(0, s.length, insideFuncUnary);
        } else {
            // compute scores and phrasal representation for leaf nodes
            for (int start = 0; start < s.length; start++) {
                insideFuncUnary.apply(start);
            }
        }


        for (int diff = 2; diff <= s.length; diff++)
        {
            final int diffFinal = diff;
            Function<Integer, Void> binaryInsideFunc =
                    new Function<Integer, Void>()
                    {
                        @Nullable
                        public Void apply(@Nullable Integer start)
                        {
                            int end = start + diffFinal;
                            log.debug("Computing Compositional inside Score for span ({}, {})", start, end);

                            final double iScore_start_end = preScores.getScore(iScores, start, end);

                            // if grammar iScores is 0, so will be comp score
                            if (iScore_start_end == 0)
                            {
                                return null;
                            }

                            for (int split = start + 1; split < end; split++)
                            {
                                INDArray child1 = s.phraseMatrix[start][split];
                                INDArray child2 = s.phraseMatrix[split][end];

                                // Compose parent (start, end) from children
                                // (start, split), (split, end)
                                s.compositionMatrix[start][end][split] =
                                        s.compositionMatrix[start][end][split].add(
                                                model.compose(child1, child2));

                                // Composition energy of parent (start,end)
                                // by children (start, split), (split, end)
                                val energy = model.energy(
                                        s.compositionMatrix[start][end][split],
                                        child1, child2);

                                double score = ((double)exp(-energy));

                                s.compositionScore[start][end][split] += score;

                                double compSplitScore = score *
                                        s.cumlCompositionScore[start][split] *
                                        s.cumlCompositionScore[split][end];

                                // Marginalize over split.
                                // This is composition score of span (start, end)
                                s.cumlCompositionScore[start][end] += compSplitScore;

                                // iSplitScore consist iscore of (start, split) and
                                // (split, end), so we just need to multiply
                                // them with cuml composition score
                                // For the binary rule, we multiply with binary composition
                                // score
                                double compISplitScore = preScores.getScore(
                                        iSplitScores, start, end, split) *
                                        compSplitScore;

                                s.compositionISplitScore[start][end][split] += compISplitScore;

                                // Composition iScore
                                // Marginalized over various splits
                                s.compositionalIScore[start][end] += compISplitScore;

                                // Phrase representation is weighted average
                                // of various split with iScores as weights
                                s.phraseMatrix[start][end] =
                                        s.phraseMatrix[start][end].add(
                                                s.compositionMatrix[start][end][split].mul(
                                                        compISplitScore));

                            }
                            // normalize weights to get them to sum to 1.
                            s.phraseMatrix[start][end] =
                                    s.phraseMatrix[start][end].div(s.compositionalIScore[start][end]);
                            return null;
                        }
                    };

            if (op.trainOp.parallel) {
                parallelizer.parallelizer(0, s.length - diff + 1, binaryInsideFunc);
            } else
            {
                for (int start = 0; start <= s.length - diff; start++)
                {
                    binaryInsideFunc.apply(start);
                }
            }
        }




    }

    /**
     * Calculate composition outside score
     */
    public void doOutsideScore(final CompositionalInsideOutsideScore s, final IInsideOutsideScore preScores) {
        final SparseMatrix oScoreWParent = preScores.getOutsideSpanWParentScore();
        for (int diff = 1; diff <= s.length; diff++) {
            final int diffFinal = diff;
            Function<Integer, Void> outsideUnaryFunc = new Function<Integer, Void>()
            {
                @Nullable
                public Void apply(@Nullable Integer start)
                {
                    int end = start + diffFinal;
                    log.debug("Computing Compositional oScore for span ({}, {})", start, end);

                    s.compositionalOScore[start][end] +=
                            preScores.getScore(oScoreWParent, start, end, end) *
                                    s.compositionScore[start][end][start];

                    // Composition oScore is calculated using the grammar outside score by
                    // multiplying cumlComposition score for expanded child and composition
                    // score for binary rule.
                    for (int parentL = 0; parentL < start; parentL++) {
                        // composition o score is composition score of parent(parentL, end)
                        // by children (parentL, start), (start, end).
                        // Then it is multiplied by cumilative score for (parentL, start)
                        // which was expanded
                        s.compositionalOScore[start][end] +=
                                preScores.getScore(oScoreWParent, start, end, parentL) *
                                        s.compositionScore[parentL][end][start] *
                                        s.cumlCompositionScore[parentL][start];
                    }

                    for (int parentR = end + 1; parentR <= s.length; parentR++) {
                        // composition o score is composition score of parent(start, parentR)
                        // by children (start, end), (end, parentR).
                        // Then it is multiplied by cumilative score for (end, parentR)
                        // which was expanded
                        s.compositionalOScore[start][end] +=
                                preScores.getScore(oScoreWParent, start, end,parentR) *
                                        s.compositionScore[start][parentR][end] *
                                        s.cumlCompositionScore[end][parentR];
                    }
                    return null;
                }
            };

            if (op.trainOp.parallel) {
                parallelizer.parallelizer(0, s.length - diff + 1, outsideUnaryFunc);
            } else {
                for (int start = 0; start + diff <= s.length; start++) {
                    outsideUnaryFunc.apply(start);
                }
            }

        }
    }

    /**
     * Calculate compositional mu score
     */
    public void doMuScore(final CompositionalInsideOutsideScore s, final IInsideOutsideScore preScores) {

        final SparseMatrix muSplitSpanScoresWParents = preScores.getMuSpanSplitScoreWParent();

        Function<Integer, Void> muUnaryFunc = new Function<Integer, Void>()
        {
            @Nullable
            public Void apply(@Nullable Integer start)
            {
                int end = start + 1;
                int split = start;
                log.debug("Computing Compositional mu Score for span ({}, {})", start, end);
                // muScore is computed as  iScore * oScore.
                // To compute compositional mu score, we need to compute
                // TODO:: Will this always be zero?

                double compSplitScore =
                        s.compositionScore[start][end][split];

                for (int parentL = 0; parentL < start; parentL++) {
                    s.compositionalMu[start][end][split] +=
                            preScores.getScore(muSplitSpanScoresWParents, start, end, split, parentL) *
                                    s.compositionScore[parentL][end][start] *
                                    s.cumlCompositionScore[parentL][start] *
                                    compSplitScore;
                }

                s.compositionalMu[start][end][split] +=
                        preScores.getScore(muSplitSpanScoresWParents, start, end, split, end) *
                                compSplitScore;


                for (int parentR = end + 1; parentR <= s.length; parentR++) {
                    s.compositionalMu[start][end][split] +=
                            preScores.getScore(muSplitSpanScoresWParents, start, end, split, parentR) *
                                    s.compositionScore[start][parentR][end] *
                                    compSplitScore *
                                    s.cumlCompositionScore[end][parentR];
                }

                return null;
            }
        };

        if (op.trainOp.parallel) {
            parallelizer.parallelizer(0, s.length, muUnaryFunc);
        } else {
            // do leaf nodes
            for (int start = 0; start < s.length; start++) {
                muUnaryFunc.apply(start);
            }
        }


        for (int diff = 2; diff <= s.length; diff++) {
            final int diffFinal = diff;
            Function<Integer, Void> muBinaryFunc =  new Function<Integer, Void>() {

                @Nullable
                public Void apply(@Nullable Integer start)
                {
                    int end = start + diffFinal;

                    log.debug("Computing Compositional mu Score for span ({}, {})", start, end);

                    for (int split = start + 1; split < end; split++) {

                        double compSplitScore =
                                s.compositionScore[start][end][split] *
                                        s.cumlCompositionScore[start][split] *
                                        s.cumlCompositionScore[split][end];

                        for (int parentL = 0; parentL < start; parentL++) {
                            s.compositionalMu[start][end][split] +=
                                    preScores.getScore(muSplitSpanScoresWParents, start, end, split,parentL) *
                                            s.compositionScore[parentL][end][start] *
                                            s.cumlCompositionScore[parentL][start] *
                                            compSplitScore;
                        }

                        s.compositionalMu[start][end][split] +=
                                preScores.getScore(muSplitSpanScoresWParents, start, end, split,end) *
                                        compSplitScore;


                        for (int parentR = end + 1; parentR <= s.length; parentR++) {
                            s.compositionalMu[start][end][split] +=
                                    preScores.getScore(muSplitSpanScoresWParents, start, end, split, parentR) *
                                            s.compositionScore[start][parentR][end] *
                                            compSplitScore *
                                            s.cumlCompositionScore[end][parentR];
                        }
                    }
                    return null;
                }
            };

            if (op.trainOp.parallel) {
                parallelizer.parallelizer(0, s.length - diff + 1, muBinaryFunc);
            } else {
                for (int start = 0; start + diff <= s.length; start++) {
                    muBinaryFunc.apply(start);
                }
            }

        }
    }

    @SneakyThrows
    public double computeCompInsideOutsideScores(CompositionalInsideOutsideScore s, IInsideOutsideScore preScores) {

        int idx = s.sentence.getIndex();
        int sz = s.sentence.size();

        log.info("Starting Compositional inside score:{}:: {}", idx, sz);
        doInsideScore(s, preScores);
        log.info("Computed Compositional inside score:{}:: {}", idx, sz);

        log.info("Starting Compositional outside score:{}:: {}", idx, sz);
        doOutsideScore(s, preScores);
        log.info("Computed Compositional outside score:{}:: {}", idx, sz);

        log.info("Starting Compositional mu score:{}:: {}", idx, sz);
        doMuScore(s, preScores);
        log.info("Computed Compositional mu score:{}:: {}", idx, sz);

        log.info("Compositional Score for sentence#{}:: {} => {}",
                idx, sz, s.getSentenceScore());

        if (op.debug) {
            log.info("Memory Size compIOScore: {}:: {}\n" +
                            "\t {} => {} MB\n" +
                            "\t {} => {} MB\n" +
                            "\t {} => {} MB\n" +
                            "\t {} => {} MB\n" +
                            "\t {} => {} MB\n" +
                            "\t {} => {} MB\n" +
                            "\t {} => {} MB\n" +
                            "\t {} => {} MB\n" +
                            "\t {} => {} MB\n" +
                            "total => {} MB",
                    idx, sz,
                    "phraseMatrix", getSize(s.phraseMatrix),
                    "compositionMatrix", getSize(s.compositionMatrix),
                    "compositionalMu", getSize(s.compositionalMu),
                    "compositionalIScore", getSize(s.compositionalIScore),
                    "compositionISplitScore", getSize(s.compositionISplitScore),
                    "compositionalOScore", getSize(s.compositionalOScore),
                    "compositionScore", getSize(s.compositionScore),
                    "cumlCompositionScore", getSize(s.cumlCompositionScore),
                    "preScores", getSize(preScores),
                    getSize(s));
        }

        return s.getSentenceScore();
    }
}
