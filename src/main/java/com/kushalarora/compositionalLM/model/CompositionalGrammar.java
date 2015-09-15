package com.kushalarora.compositionalLM.model;

import com.kushalarora.compositionalLM.lang.IInsideOutsideScore;
import com.kushalarora.compositionalLM.lang.Sentence;
import com.kushalarora.compositionalLM.lang.Word;
import com.kushalarora.compositionalLM.options.Options;
import com.kushalarora.compositionalLM.utils.ObjectSizeFetcher;
import lombok.Getter;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.ujmp.core.SparseMatrix;

import java.io.Serializable;
import java.util.List;

import static com.kushalarora.compositionalLM.utils.ObjectSizeFetcher.getSize;
import static java.lang.Math.exp;

/**
 * Created by karora on 6/21/15.
 */
@Slf4j
@Getter
public class CompositionalGrammar implements Serializable {
    private final Options op;
    private Model model;

    public CompositionalGrammar(final Model model, final Options op) {
        this.model = model;
        this.op = op;
    }

    public class CompositionalInsideOutsideScore {
        // Averaged representation of phrases in sentence
        private transient INDArray[][] phraseMatrix;

        // composition matrix for all possible phrases
        // contains multiple representation for each
        // phrase originating from different split
        private transient INDArray[][][] compositionMatrix;

        // extended mu to included compositional score
        private transient double[][][] compositionalMu;

        // extended inside score with compositional score
        private transient double[][] compositionalIScore;

        private transient double[][][] compositionISplitScore;

        // extended outside score extended with compositional score
        private transient double[][] compositionalOScore;

        // score for each phrase composition.
        private transient double[][][] compositionScore;

        // sum of scores all possible composition of a phrase.
        // marginalization over split
        private transient double[][] cumlCompositionScore;

        // current maximum length that we can handle
        private transient int arraySize;

        private int myMaxLength;

        private Sentence sentence;

        private int length;

        private IInsideOutsideScore preScores;


        /**
         * Clear all matrices.
         */
        public void clearMatrices() {
            // IndArray [start][end]
            phraseMatrix = null;

            // IndArray [start][end][split]
            compositionMatrix = null;

            // double [start][end]
            compositionalIScore = null;

            // double  [start][end][split]
            compositionISplitScore = null;

            // double [start][end]
            compositionalOScore = null;

            // double[start][end][split]
            compositionalMu = null;

            // double [start][end][split]
            compositionScore = null;

            // double [start][end]
            cumlCompositionScore = null;

            arraySize = 0;
        }

        /**
         * Create matrix if the length is greater than the previous
         * matrices else use the same. Initialize them by wiping out
         */
        public void considerCreatingMatrices() {
            try {
                createMatrices(length);
            } catch (Exception e) {
                log.error("Unable to create array of length {}",
                        length);
                throw new RuntimeException(e);
            }
        }

        /**
         * Allocate memory to matrices
         *
         * @param length length of the current sentence
         */
        // TODO:: Either remove length as argument here
        // and in considerCreatingMatrices or add dimension as argument
        private void createMatrices(int length) {
            clearMatrices();
            log.info("Creating Compositional matrices for length {}: {}", length, sentence.getIndex());
            int dim = model.getDimensions();
            phraseMatrix = new INDArray[length][length + 1];

            compositionMatrix = new INDArray[length][length + 1][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    phraseMatrix[start][end] = Nd4j.create(dim, 1);
                    compositionMatrix[start][end] = new INDArray[length];
                    for (int split = start; split < end; split++) {
                        compositionMatrix[start][end][split] = Nd4j.create(dim, 1);
                    }
                }
            }

            compositionalMu = new double[length][length + 1][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    compositionalMu[start][end] = new double[length];
                }
            }

            compositionalIScore = new double[length][length + 1];
            compositionalOScore = new double[length][length + 1];
            compositionISplitScore = new double[length][length + 1][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    compositionISplitScore[start][end] = new double[length];
                }
            }

            cumlCompositionScore = new double[length][length + 1];
            compositionScore = new double[length][length + 1][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    compositionScore[start][end] = new double[length];
                }
            }
        }

        /**
         * Initialize all matrices to zeros
         */
        public void initializeMatrices() {
            log.info("Initializing Compositional matrices:{}", sentence.getIndex());
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {

                    for (int d = 0; d < model.getDimensions(); d++) {
                        phraseMatrix[start][end].putScalar(d, 0);
                    }

                    compositionalIScore[start][end] = 0;
                    compositionalOScore[start][end] = 0;
                    cumlCompositionScore[start][end] = 0;

                    compositionISplitScore[start][end][start] = 0f;
                    compositionalMu[start][end][start] = 0f;

                    compositionScore[start][end][start] = 0;
                    for (int split = start; split < end; split++) {
                        compositionScore[start][end][split] = 0;
                        compositionalMu[start][end][split] = 0f;
                        compositionISplitScore[start][end][split] = 0f;
                        for (int d = 0; d < model.getDimensions(); d++) {
                            compositionMatrix[start][end][split].putScalar(d, 0);
                        }
                    }
                }
            }
        }

        /**
         * Calculate compositional iScore, composition scores,
         * cumulative composition score and phrase and composition matrix.
         */
        public void doInsideScore() {

            SparseMatrix iScores = preScores.getInsideSpanProb();
            SparseMatrix iSplitScores = preScores.getInsideSpanSplitProb();

            // compute scores and phrasal representation for leaf nodes
            for (int start = 0; start < length; start++) {
                int end = start + 1;
                int split = start;
                log.debug("Computing Compositional inside Score for span ({}, {})", start, end);

                // Set phrase for word sentence[start]
                phraseMatrix[start][end] = phraseMatrix[start][end].add(
                        model.word2vec(sentence.get(start)));

                compositionMatrix[start][end][split] =
                        compositionMatrix[start][end][split].add(
                                phraseMatrix[start][end]);

                // For leaf nodes, the energy of the node is
                // a function of phrase representation
                val energy = model.energy(phraseMatrix[start][end]);

                // un-normalized prob or score, is negative exp of energy
                double score = ((double) exp(-energy));

                cumlCompositionScore[start][end] += score;
                compositionScore[start][end][split] += score;

                // comp iScore for a leaf node is iScore of span (start, start + 1)
                // into composition score of (start, start + 1)
                // This is like re-writing unary rule a=>w_i
                double iScore_start_end = preScores.getScore(iScores, start, end);
                compositionalIScore[start][end] += score * iScore_start_end;
                compositionISplitScore[start][end][split] += score * iScore_start_end;
            }


            for (int diff = 2; diff <= length; diff++) {
                for (int start = 0; start <= length - diff; start++) {
                    int end = start + diff;
                    log.debug("Computing Compositional inside Score for span ({}, {})", start, end);

                    double iScore_start_end = preScores.getScore(iScores, start, end);
                    // if grammar iScores is 0, so will be comp score
                    if (iScore_start_end == 0) {
                        throw new RuntimeException(
                                String.format("Span iScore[%d][%d] == zero", start, end));
                    }

                    for (int split = start + 1; split < end; split++) {
                        INDArray child1 = phraseMatrix[start][split];
                        INDArray child2 = phraseMatrix[split][end];

                        // Compose parent (start, end) from children
                        // (start, split), (split, end)
                        compositionMatrix[start][end][split] =
                                compositionMatrix[start][end][split].add(
                                        model.compose(child1, child2));

                        // Composition energy of parent (start,end)
                        // by children (start, split), (split, end)
                        val energy = model.energy(
                                compositionMatrix[start][end][split],
                                child1, child2);

                        double score = ((double) exp(-energy));

                        compositionScore[start][end][split] += score;

                        double compSplitScore = score *
                                cumlCompositionScore[start][split] *
                                cumlCompositionScore[split][end];

                        // Marginalize over split.
                        // This is composition score of span (start, end)
                        cumlCompositionScore[start][end] += compSplitScore;

                        // iSplitScore consist iscore of (start, split) and
                        // (split, end), so we just need to multiply
                        // them with cuml composition score
                        // For the binary rule, we multiply with binary composition
                        // score
                        double compISplitScore = preScores.getScore(
                                iSplitScores, start, end, split) *
                                compSplitScore;

                        compositionISplitScore[start][end][split] += compISplitScore;

                        // Composition iScore
                        // Marginalized over various splits
                        compositionalIScore[start][end] += compISplitScore;

                        // Phrase representation is weighted average
                        // of various split with iScores as weights
                        phraseMatrix[start][end] =
                                phraseMatrix[start][end].add(
                                        compositionMatrix[start][end][split].mul(
                                                compISplitScore));

                    }
                    // normalize weights to get them to sum to 1.
                    phraseMatrix[start][end] =
                            phraseMatrix[start][end].div(compositionalIScore[start][end]);
                }
            }
        }

        /**
         * Calculate composition outside score
         */
        public void doOutsideScore() {
            SparseMatrix oScoreWParent = preScores.getOutsideSpanWParentScore();
            for (int diff = 1; diff <= length; diff++) {
                for (int start = 0; start + diff <= length; start++) {
                    int end = start + diff;
                    log.debug("Computing Compositional oScore for span ({}, {})", start, end);

                    compositionalOScore[start][end] +=
                            preScores.getScore(oScoreWParent, start, end, end) *
                                    compositionScore[start][end][start];

                    // Composition oScore is calculated using the grammar outside score by
                    // multiplying cumlComposition score for expanded child and composition
                    // score for binary rule.
                    for (int parentL = 0; parentL < start; parentL++) {
                        // composition o score is composition score of parent(parentL, end)
                        // by children (parentL, start), (start, end).
                        // Then it is multiplied by cumilative score for (parentL, start)
                        // which was expanded
                        compositionalOScore[start][end] +=
                                preScores.getScore(oScoreWParent, start, end, parentL) *
                                        compositionScore[parentL][end][start] *
                                        cumlCompositionScore[parentL][start];
                    }

                    for (int parentR = end + 1; parentR <= length; parentR++) {
                        // composition o score is composition score of parent(start, parentR)
                        // by children (start, end), (end, parentR).
                        // Then it is multiplied by cumilative score for (end, parentR)
                        // which was expanded
                        compositionalOScore[start][end] +=
                                preScores.getScore(oScoreWParent, start, end,parentR) *
                                        compositionScore[start][parentR][end] *
                                        cumlCompositionScore[end][parentR];
                    }
                }
            }
        }

        /**
         * Calculate compositional mu score
         */
        public void doMuScore() {

            SparseMatrix muSplitSpanScoresWParents = preScores.getMuSpanSplitScoreWParent();
            // do leaf nodes
            for (int start = 0; start < length; start++) {
                int end = start + 1;
                int split = start;
                log.debug("Computing Compositional mu Score for span ({}, {})", start, end);
                // muScore is computed as  iScore * oScore.
                // To compute compositional mu score, we need to compute
                // TODO:: Will this always be zero?

                double compSplitScore =
                        compositionScore[start][end][split];

                for (int parentL = 0; parentL < start; parentL++) {
                    compositionalMu[start][end][split] +=
                            preScores.getScore(muSplitSpanScoresWParents, start, end, split, parentL) *
                                    compositionScore[parentL][end][start] *
                                    cumlCompositionScore[parentL][start] *
                                    compSplitScore;
                }

                compositionalMu[start][end][split] +=
                        preScores.getScore(muSplitSpanScoresWParents, start, end, split, end) *
                                compSplitScore;


                for (int parentR = end + 1; parentR <= length; parentR++) {
                    compositionalMu[start][end][split] +=
                            preScores.getScore(muSplitSpanScoresWParents, start, end, split, parentR) *
                                    compositionScore[start][parentR][end] *
                                    compSplitScore *
                                    cumlCompositionScore[end][parentR];
                }

            }
            for (int diff = 2; diff <= length; diff++) {
                for (int start = 0; start + diff <= length; start++) {
                    int end = start + diff;

                    log.debug("Computing Compositional mu Score for span ({}, {})", start, end);

                    for (int split = start + 1; split < end; split++) {

                        double compSplitScore =
                                compositionScore[start][end][split] *
                                        cumlCompositionScore[start][split] *
                                        cumlCompositionScore[split][end];

                        for (int parentL = 0; parentL < start; parentL++) {
                            compositionalMu[start][end][split] +=
                                    preScores.getScore(muSplitSpanScoresWParents, start, end, split,parentL) *
                                            compositionScore[parentL][end][start] *
                                            cumlCompositionScore[parentL][start] *
                                            compSplitScore;
                        }

                        compositionalMu[start][end][split] +=
                                preScores.getScore(muSplitSpanScoresWParents, start, end, split,end) *
                                        compSplitScore;


                        for (int parentR = end + 1; parentR <= length; parentR++) {
                            compositionalMu[start][end][split] +=
                                    preScores.getScore(muSplitSpanScoresWParents, start, end, split, parentR) *
                                            compositionScore[start][parentR][end] *
                                            compSplitScore *
                                            cumlCompositionScore[end][parentR];
                        }
                    }
                }
            }
        }

        public CompositionalInsideOutsideScore(Sentence sentence, IInsideOutsideScore preScore) {
            arraySize = 0;
            myMaxLength = op.grammarOp.maxLength;
            this.sentence = sentence;
            length = sentence.size();
            this.preScores = preScore;
        }

        public double[][] getInsideSpanProb() {
            return compositionalIScore;
        }

        public double[][][] getMuScore() {
            return compositionalMu;
        }

        public INDArray[][][] getCompositionMatrix() {
            return compositionMatrix;
        }

        public INDArray[][] getPhraseMatrix() {
            return phraseMatrix;
        }

        public double[][][] getCompositionISplitScore() {
            return compositionISplitScore;
        }


        /**
         * Score is length normalized
         *
         * @return score
         */
        public double getSentenceScore() {
            return exp(Math.log(compositionalIScore[0][length]) / length);
        }

        @SneakyThrows
        public double computeCompInsideOutsideScores() {
            considerCreatingMatrices();
            initializeMatrices();

            log.info("Starting Compositional inside score:{}:: {}", sentence.getIndex(), sentence.size());
            doInsideScore();
            log.info("Computed Compositional inside score:{}:: {}", sentence.getIndex(), sentence.size());

            log.info("Starting Compositional outside score:{}:: {}", sentence.getIndex(), sentence.size());
            doOutsideScore();
            log.info("Computed Compositional outside score:{}:: {}", sentence.getIndex(), sentence.size());

            log.info("Starting Compositional mu score:{}:: {}", sentence.getIndex(), sentence.size());
            doMuScore();
            log.info("Computed Compositional mu score:{}:: {}", sentence.getIndex(), sentence.size());

            log.info("Compositional Score for sentence#{}:: {} => {}",
                    sentence.getIndex(),sentence.size(), getSentenceScore());

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
                        sentence.getIndex(), sentence.size(),
                        "phraseMatrix", getSize(phraseMatrix),
                        "compositionMatrix", getSize(compositionMatrix),
                        "compositionalMu", getSize(compositionalMu),
                        "compositionalIScore", getSize(compositionalIScore),
                        "compositionISplitScore", getSize(compositionISplitScore),
                        "compositionalOScore", getSize(compositionalOScore),
                        "compositionScore", getSize(compositionScore),
                        "cumlCompositionScore", getSize(cumlCompositionScore),
                        "preScores", getSize(preScores),
                        getSize(this));
            }

            return getSentenceScore();
        }
    }


    public void parse(List<Word> sentence) {

    }

    public void nbest(List<List<Word>> decodings) {

    }

    public CompositionalInsideOutsideScore getScore(
            Sentence sentence, IInsideOutsideScore preScores) {
        return new CompositionalInsideOutsideScore(sentence, preScores);
    }

    public CompositionalInsideOutsideScore computeScore(Sentence sentence,
                                                        IInsideOutsideScore preScore) {
        CompositionalInsideOutsideScore score = getScore(sentence, preScore);
        score.computeCompInsideOutsideScores();
        return score;
    }
}
