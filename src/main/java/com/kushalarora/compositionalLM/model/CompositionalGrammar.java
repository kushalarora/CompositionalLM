package com.kushalarora.compositionalLM.model;

import com.kushalarora.compositionalLM.lang.IInsideOutsideScorer;
import com.kushalarora.compositionalLM.lang.Word;
import com.kushalarora.compositionalLM.options.Options;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.util.List;

import static java.lang.Math.exp;

/**
 * Created by karora on 6/21/15.
 */
@Slf4j
public class CompositionalGrammar implements Serializable {
    private final Options op;
    private Model model;

    public CompositionalGrammar(final Model model, final Options op) {
        this.model = model;
        this.op = op;
//        cache =
//        CacheBuilder.newBuilder()
//                .maximumWeight(50)
//                .weigher(new Weigher<List<Word>, IInsideOutsideScorer>() {
//                    public int weigh(List<Word> words, IInsideOutsideScorer iInsideOutsideScorer) {
//                        return  words.size();
//                    }
//                })
//                .build(new CacheLoader<List<Word>, IInsideOutsideScorer>() {
//                    @Override
//                    public IInsideOutsideScorer load(List<Word> sentence) throws Exception {
//                        return model.getGrammar().computeScore(sentence);
//                    }
//                });
    }

  //  LoadingCache<List<Word>, IInsideOutsideScorer> cache;


    public class CompositionalInsideOutsideScorer {
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
        }

        /**
         * Create matrix if the length is greater than the previous
         * matrices else use the same. Initialize them by wiping out
         *
         * @param length length of the current sentence
         */
        public void considerCreatingMatrices(int length) {
            if (length > op.grammarOp.maxLength ||
                    // myMaxLength if greater than zero,
                    // then it is max memory size
                    length >= myMaxLength) {
                throw new OutOfMemoryError("Refusal to create such large arrays.");
            } else if (arraySize < length) {

                try {
                    createMatrices(length);
                    arraySize = length;
                } catch (Exception e) {
                    log.error("Unable to create array of length {}. Reverting to size: {}",
                            length, arraySize);
                    myMaxLength = length;
                    createMatrices(arraySize);
                    throw new RuntimeException(e);
                }
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
            log.info("Creating Compositional matrices for length {}", length);
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
         *
         * @param length length of the current sentence
         */
        public void initializeMatrices(int length) {
            log.info("Initializing Compositional matrices");
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
        public void doInsideScore(List<Word> sentence,
                                  int length, IInsideOutsideScorer preScores) {

            double[][] iScores = preScores.getInsideSpanProb();
            double[][][] iSplitScores = preScores.getInsideSpanSplitProb();

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
                compositionalIScore[start][end] += score * iScores[start][end];
                compositionISplitScore[start][end][split] += score * iScores[start][end];
            }


            for (int diff = 2; diff <= length; diff++) {
                for (int start = 0; start <= length - diff; start++) {
                    int end = start + diff;
                    log.debug("Computing Compositional inside Score for span ({}, {})", start, end);

                    // if grammar iScores is 0, so will be comp score
                    if (iScores[start][end] == 0) {
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
                        double compISplitScore = iSplitScores[start][end][split] * compSplitScore;

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
                    // normalize weights to get them sum to 1.
                    phraseMatrix[start][end] =
                            phraseMatrix[start][end].div(compositionalIScore[start][end]);
                }
            }
        }

        /**
         * Calculate composition outside score
         */
        public void doOutsideScore(int length, IInsideOutsideScorer preScores) {
            double[][][] oScoreWParent = preScores.getOutsideSpanWParentScore();
            for (int diff = 1; diff <= length; diff++) {
                for (int start = 0; start + diff <= length; start++) {
                    int end = start + diff;
                    log.debug("Computing Compositional oScore for span ({}, {})", start, end);

                    compositionalOScore[start][end] +=
                            oScoreWParent[start][end][end] *
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
                                oScoreWParent[start][end][parentL] *
                                        compositionScore[parentL][end][start] *
                                        cumlCompositionScore[parentL][start];
                    }

                    for (int parentR = end + 1; parentR <= length; parentR++) {
                        // composition o score is composition score of parent(start, parentR)
                        // by children (start, end), (end, parentR).
                        // Then it is multiplied by cumilative score for (end, parentR)
                        // which was expanded
                        compositionalOScore[start][end] +=
                                oScoreWParent[start][end][parentR] *
                                        compositionScore[start][parentR][end] *
                                        cumlCompositionScore[end][parentR];
                    }
                }
            }
        }

        /**
         * Calculate compositional mu score
         */
        public void doMuScore(int length, IInsideOutsideScorer preScores) {

            double[][][][] muSplitSpanScoresWParents = preScores.getMuSpanScoreWParent();
            // do leaf nodes
            for (int start = 0; start < length; start++) {
                int end = start + 1;
                int split = start;
                log.debug("Computing Compositional mu Score for span ({}, {})", start, end);
                // muScore is computed as  iScore * oScore.
                // To compute compositional mu score, we need to compute
                // TODO:: Will this always be zero?

                double compSplitScore =
                        compositionScore[start][end][split] *
                                cumlCompositionScore[start][split] *
                                cumlCompositionScore[split][end];

                for (int parentL = 0; parentL < start; parentL++) {
                    compositionalMu[start][end][split] +=
                            muSplitSpanScoresWParents[start][end][split][parentL] *
                                    compositionScore[parentL][end][start] *
                                    cumlCompositionScore[parentL][start] *
                                    compSplitScore;
                }

                compositionalMu[start][end][split] +=
                        muSplitSpanScoresWParents[start][end][split][end] *
                                compSplitScore;


                for (int parentR = end + 1; parentR <= length; parentR++) {
                    compositionalMu[start][end][split] +=
                            muSplitSpanScoresWParents[start][end][split][parentR] *
                                    compositionScore[start][parentR][end] *
                                    compSplitScore *
                                    cumlCompositionScore[end][parentR];
                }

            }
            for (int diff = 1; diff <= length; diff++) {
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
                                    muSplitSpanScoresWParents[start][end][split][parentL] *
                                            compositionScore[parentL][end][start] *
                                            cumlCompositionScore[parentL][start] *
                                            compSplitScore;
                        }

                        compositionalMu[start][end][split] +=
                                muSplitSpanScoresWParents[start][end][split][end] *
                                        compSplitScore;


                        for (int parentR = end + 1; parentR <= length; parentR++) {
                            compositionalMu[start][end][split] +=
                                    muSplitSpanScoresWParents[start][end][split][parentR] *
                                            compositionScore[start][parentR][end] *
                                            compSplitScore *
                                            cumlCompositionScore[end][parentR];
                        }
                    }
                }
            }
        }

        public CompositionalInsideOutsideScorer() {
            arraySize = 0;
            myMaxLength = Integer.MAX_VALUE;
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


        @SneakyThrows
        public double computeCompInsideOutsideScores(List<Word> sentence) {
            int length = sentence.size();
            considerCreatingMatrices(length);
            initializeMatrices(length);

            // IMPORTANT: Length must be calculated before this
            IInsideOutsideScorer preScorer =
                 ///   cache.get(sentence);
                 model.getGrammar().computeScore(sentence);

            log.info("Starting Computational inside score");
            doInsideScore(sentence, length, preScorer);
            log.info("Computed Computational inside score");

            log.info("Starting Computational outside score");
            doOutsideScore(length, preScorer);
            log.info("Computed Computational outside score");

            log.info("Starting Computational mu score");
            doMuScore(length, preScorer);
            log.info("Starting Computational mu score");

            return compositionalIScore[0][length];
        }
    }


    public void parse(List<Word> sentence) {

    }

    public void nbest(List<List<Word>> decodings) {

    }

    public CompositionalInsideOutsideScorer getScorer() {
        return new CompositionalInsideOutsideScorer();
    }

}
