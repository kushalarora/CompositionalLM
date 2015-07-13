package com.kushalarora.compositionalLM.model;

import com.kushalarora.compositionalLM.lang.IGrammar;
import com.kushalarora.compositionalLM.lang.IInsideOutsideScores;
import com.kushalarora.compositionalLM.lang.Word;
import com.kushalarora.compositionalLM.options.GrammarOptions;
import com.kushalarora.compositionalLM.options.Options;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

import static java.lang.Math.exp;

/**
 * Created by karora on 6/21/15.
 */
@Slf4j
public class CompositionalGrammar {
    private final Options op;
    List<Word> sentence;
    private Model model;
    private int myMaxLength;

    public CompositionalGrammar(Model model, Options op) {
        this.model = model;
        this.op = op;
        myMaxLength = Integer.MAX_VALUE;
    }

    public class CompositionalInsideOutsideScorer {
        private final int length;
        // Averaged representation of phrases in sentence
        private transient INDArray[][] phraseMatrix;

        // composition matrix for all possible phrases
        // contains multiple representation for each
        // phrase originating from different split
        private transient INDArray[][][] compositionMatrix;

        // extended mu to included compositional score
        private transient float[][][] compositionalMu;

        // extended inside score with compositional score
        private transient float[][] compositionalIScore;

        private transient float[][][] compositionISplitScore;

        // extended outside score extended with compositional score
        private transient float[][] compositionalOScore;

        // score for each phrase composition.
        private transient float[][][] compositionScore;

        // sum of scores all possible composition of a phrase.
        // marginalization over split
        private transient float[][] cumlCompositionScore;

        // current maximum length that we can handle
        private transient int arraySize;

        // current sentence being processed
        List<Word> sentence;

        // Inside outside score object for grammar
        // being used
        IInsideOutsideScores preScores;

        /**
         * Clear all matrices.
         */
        private void clearMatrices() {
            // IndArray [start][end]
            phraseMatrix = null;

            // IndArray [start][end][split]
            compositionMatrix = null;

            // float [start][end]
            compositionalIScore = null;

            // float  [start][end][split]
            compositionISplitScore = null;

            // float [start][end]
            compositionalOScore = null;

            // float[start][end][split]
            compositionalMu = null;

            // float [start][end][split]
            compositionScore = null;

            // float [start][end]
            cumlCompositionScore = null;
        }

        /**
         * Create matrix if the length is greater than the previous
         * matrices else use the same. Initialize them by wiping out
         *
         * @param length length of the current sentence
         */
        private void considerCreatingMatrices(int length) {
            if (length > GrammarOptions.maxLength ||
                    // myMaxLength if greater than zero,
                    // then it is max memory size
                    length >= myMaxLength) {
                throw new OutOfMemoryError("Refusal to create such large arrays.");
            } else if (arraySize < length) {
                clearMatrices();
                try {
                    createMatrices(length);
                    arraySize = length;
                } catch (Exception e) {
                    log.error("Unable to create array of length {}. Reverting to size: {}",
                            length, arraySize);
                    myMaxLength = length;
                    createMatrices(arraySize);
                }
            }
            initializeMatrices(length);
        }

        /**
         * Allocate memory to matrices
         *
         * @param length length of the current sentence
         */
        // TODO:: Either remove length as argument here
        // and in considerCreatingMatrices or add dimension as argument
        public void createMatrices(int length) {
            int dim = model.getDimensions();
            phraseMatrix = new INDArray[length][length + 1];

            compositionMatrix = new INDArray[length][length + 1][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    phraseMatrix[start][end] = Nd4j.create(dim, 1);
                    compositionMatrix[start][end] = new INDArray[length];
                    for (int split = start + 1; split < end; split++) {
                        compositionMatrix[start][end][split] = Nd4j.create(dim, 1);
                    }
                }
            }

            compositionalMu = new float[length][length + 1][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    compositionalMu[start][end] = new float[length];
                }
            }

            compositionalIScore = new float[length][length + 1];
            compositionalOScore = new float[length][length + 1];
            compositionISplitScore = new float[length][length + 1][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    compositionISplitScore[start][end] = new float[length];
                }
            }

            cumlCompositionScore = new float[length][length + 1];
            compositionScore = new float[length][length + 1][];
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {
                    compositionScore[start][end] = new float[length];
                }
            }
        }

        /**
         * Initialize all matrices to zeros
         *
         * @param length length of the current sentence
         */
        public void initializeMatrices(int length) {
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

                    for (int split = start + 1; split < end; split++) {
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

            float[][] iScores = preScores.getInsideSpanProb();
            float[][][] iSplitScores = preScores.getInsideSpanSplitProb();

            // compute scores and phrasal representation for leaf nodes
            for (int start = 0; start < length; start++) {
                int end = start + 1;
                int split = start;

                // Set phrase for word sentence[start]
                phraseMatrix[start][end] = model.word2vec(sentence.get(start));

                // For leaf nodes, the energy of the node is
                // a function of phrase representation
                val energy = model.energy(phraseMatrix[start][end]);

                // un-normalized prob or score, is negative exp of energy
                float score = ((float) exp(-energy));

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
                        compositionMatrix[start][end][split] = model.compose(
                                child1, child2);

                        // Composition energy of parent (start,end)
                        // by children (start, split), (split, end)
                        val energy = model.energy(
                                compositionMatrix[start][end][split],
                                child1, child2);
                        float score = ((float) exp(-energy));

                        compositionScore[start][end][split] += score;

                        // Marginalize over split.
                        // This is composition score of span (start, end)
                        cumlCompositionScore[start][end] += score;


                        float iSplitScore = iSplitScores[split][end][split];

                        // iSplitScore consist iscore of (start, split) and
                        // (split, end), so we just need to multiply
                        // them with cuml composition score
                        // For the binary rule, we multiply with binary composition
                        // score
                        float compISplitScore = iSplitScore * score *
                                cumlCompositionScore[start][split] *
                                cumlCompositionScore[split][end];

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
        public void doOutsideScore() {
            float[][][] oScoreWParent = preScores.getOutsideSpanWParentScore();
            for (int diff = 1; diff <= length; diff++) {
                for (int start = 0; start + diff <= length; start++) {
                    int end = start + diff;

                    // Composition oScore is calculated using the grammar outside score by
                    // multiplying cumlComposition score for expanded child and composition
                    // score for binary rule.
                    for (int parentL = 0; parentL < start; parentL++) {
                        // composition o score is composition score of parent(parentL, end)
                        // by children (parentL, start), (start, end).
                        // Then it is multiplied by cumilative score for (parentL, start)
                        // which was expanded
                        compositionalOScore[start][end] += compositionScore[parentL][end][start] *
                                cumlCompositionScore[parentL][start] * oScoreWParent[start][end][parentL];
                    }

                    for (int parentR = end; end != length && parentR <= length; parentR++) {
                        // composition o score is composition score of parent(start, parentR)
                        // by children (start, end), (end, parentR).
                        // Then it is multiplied by cumilative score for (end, parentR)
                        // which was expanded
                        compositionalOScore[start][end] += compositionScore[start][parentR][end] *
                                cumlCompositionScore[end][parentR] * oScoreWParent[start][end][parentR];
                    }
                }
            }
        }

        /**
         * Calculate compositional mu score
         */
        public void doMuScore() {
            float[][][][] muSplitSpanScoresWParents = preScores.getMuSpanScoreWParent();
            // do leaf nodes
            for (int start = 0; start < length; start++) {
                int end = start + 1;
                int split = start;

                // muScore is computed as  iScore * oScore.
                // To compute compositional mu score, we need to compute
                // TODO:: Will this always be zero?
                for (int parentL = 0; parentL < start; parentL++) {
                    compositionalMu[start][end][split] +=
                            muSplitSpanScoresWParents[start][end][split][parentL] *
                                    cumlCompositionScore[start][end] *
                                    compositionScore[parentL][end][start] *
                                    cumlCompositionScore[parentL][start];
                }
                for (int parentR = end; parentR <= length; parentR++) {
                    try {
                        compositionalMu[start][end][split] +=
                                muSplitSpanScoresWParents[start][end][split][parentR] *
                                        cumlCompositionScore[start][end] *
                                        compositionScore[start][parentR][end] *
                                        cumlCompositionScore[end][parentR];
                    } catch (Exception e) {
                        log.error("Nothing");
                    }
                }

            }
            for (int diff = 2; diff <= length; diff++) {
                for (int start = 0; start + diff <= length; start++) {
                    int end = start + diff;
                    for (int split = start + 1; split < end; split++) {

                        for (int parentL = 0; parentL < start; parentL++) {
                            compositionalMu[start][end][split] +=
                                    muSplitSpanScoresWParents[start][end][split][parentL] *
                                            cumlCompositionScore[start][end] *
                                            compositionScore[parentL][end][start] *
                                            cumlCompositionScore[parentL][start];
                        }
                        for (int parentR = end; end != length && parentR <= length; parentR++) {
                            compositionalMu[start][end][split] +=
                                    muSplitSpanScoresWParents[start][end][split][parentR] *
                                            cumlCompositionScore[start][end] *
                                            compositionScore[start][parentR][end] *
                                            cumlCompositionScore[end][parentR];
                        }
                    }
                }
            }
        }

        public CompositionalInsideOutsideScorer(List<Word> sentence) {
            arraySize = 0;
            this.sentence = sentence;
            length = sentence.size();
            preScores = model.getGrammar().computeInsideOutsideProb(sentence);
        }

        public float[][] getInsideSpanProb() {
            return compositionalIScore;
        }

        public float[][][] getMuScore() {
            return compositionalMu;
        }

        public INDArray[][][] getCompositionMatrix() {
            return compositionMatrix;
        }

        public INDArray[][] getPhraseMatrix() {
            return phraseMatrix;
        }

        public float[][][] getCompositionISplitScore() {
            return compositionISplitScore;
        }

        public List<Word> getCurrentSentence() {
            return sentence;
        }

        public float computeCompInsideOutsideScores() {
            considerCreatingMatrices(length);
            initializeMatrices(length);
            doInsideScore();
            doOutsideScore();
            doMuScore();
            return compositionalIScore[0][length];
        }
    }


    public void train(List<Word> sentence) {


        float previousScore = Float.POSITIVE_INFINITY;
        //Step 1:  Calculate priors \pi, \beta and \muSpanSplitScore

        // Step 2: Optimize
        // Step 2a Initialize matrices to be used in computation.
        val scorer = new CompositionalInsideOutsideScorer(sentence);
        int iterCount = 0;
        // While iter or tolerance
        // TODO:: Add condition for stopping optimization
        do {
            //Step 2b:  Calculate scorer for the sentence
            // Step 2ba: Get continuous vectors for word and Initialize the leaf nodes

            // Step 2bb: Iterate over inside scorer and build
            // compositional matrix and phrasal representation matrix
            // calculate posterior inside, outside probability and
            // muSpanSplitScore.
            float score = scorer.computeCompInsideOutsideScores();
            log.info("score iter {}: {}", iterCount++, score);

            // Step 2b:
            // Step 2b1: If scorer diff from previous scorer less than tolerance
            // return
            // Step 2b2: Else
            // Step 2b2a: Calculate derivatives

            // Pass compositionalMu to derivate computation functions
            // Let them handle derivative computation
            // Step 2b2b: Do SGD on params
            // Figure out how to do SGD
        } while (iterCount < 10);     // add threshold or tolerance condition
    }

    public void parse(List<Word> sentence) {

    }

    public void nbest(List<List<Word>> decodings) {

    }

    public CompositionalInsideOutsideScorer getScorer(List<Word> sentence) {
        return new CompositionalInsideOutsideScorer(sentence);
    }

}
