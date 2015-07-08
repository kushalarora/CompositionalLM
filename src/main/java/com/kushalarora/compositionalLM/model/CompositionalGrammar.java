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
    private IGrammar grammar;
    private int myMaxLength;

    public CompositionalGrammar(IGrammar grammar, Model model, Options op) {
        this.model = model;
        this.grammar = grammar;
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

        List<Word> sentence;

        // Inside outside score object for grammar
        // being used
        IInsideOutsideScores preScores;

        private void clearMatrices() {
            phraseMatrix = null;
            compositionMatrix = null;

            compositionalIScore = null;
            compositionISplitScore = null;
            compositionalOScore = null;
            compositionalMu = null;

            compositionScore = null;
            cumlCompositionScore = null;
        }

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
                    createMatrices(arraySize);
                }
            }
            initializeMatrices(length);
        }

        public void createMatrices(int length) {
            int dim = model.params.getDimensions();
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

        public void initializeMatrices(int length) {
            for (int start = 0; start < length; start++) {
                for (int end = start + 1; end <= length; end++) {

                    for (int d = 0; d < model.params.getDimensions(); d++) {
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
                        for (int d = 0; d < model.params.getDimensions(); d++) {
                            compositionMatrix[start][end][split].putScalar(d, 0);
                        }
                    }
                }
            }
        }

        public void doInsideScore() {

            float[][] iScores = preScores.getInsideSpanProb();
            float[][][] iSplitScores = preScores.getInsideSpanSplitProb();

            // compute scores and phrasal representation for leaf nodes
            for (int start = 0; start < length; start++) {
                int end = start + 1;
                phraseMatrix[start][end] = model.word2vec(sentence.get(start));

                val energy = model.energy(phraseMatrix[start][end]);
                float score = ((float) exp(-energy));

                compositionScore[start][end][start] = score;
                cumlCompositionScore[start][end] = score;
                compositionalIScore[start][end] = score * iScores[start][end];
                compositionISplitScore[start][end][start] = score * iScores[start][end];
            }


            for (int diff = 2; diff < length; diff++) {
                for (int start = 0; start <= length - diff; start++) {
                    int end = start + diff;
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

                        compositionScore[start][end][split] = score;

                        // Marginalize over split.
                        // This is composition score of span (start, end)
                        cumlCompositionScore[start][end] += score;


                        float iSplitScore = iSplitScores[split][end][split];

                        // composition iScore for span (start, end)
                        // from children (start, split), (split, end)
                        // iScore is current split * iScore of right child
                        // * iScore of left child
                        float compISplitScore = iSplitScore * score *
                                compositionalIScore[start][split] *
                                compositionalIScore[split][end];

                        compositionISplitScore[start][end][split] = compISplitScore;

                        // Phrase representation is weighted average
                        // of various split with iScores as weights
                        phraseMatrix[start][end] =
                                phraseMatrix[start][end].add(
                                        compositionMatrix[start][end][split].mul(
                                                compISplitScore));


                        // Composition iScore
                        // Marginalized over various splits
                        compositionalIScore[start][end] += compISplitScore;
                    }

                    // normalize weights to get them sum to 1.
                    phraseMatrix[start][end].div(compositionalIScore[start][end]);
                }
            }
        }

        public void doOutsideScore() {
            float[][][] oScoreWParent = preScores.getOutsideSpanWParentScore();
            for (int diff = length; diff >= 1; diff--) {
                for (int start = 0; start + diff <= length; start++) {
                    int end = start + diff;

                    for (int parentL = 0; parentL < start; parentL++) {
                        compositionalOScore[start][end] += compositionScore[parentL][end][start] *
                                cumlCompositionScore[start][end] * oScoreWParent[start][end][parentL];
                    }

                    for (int parentR = end; parentR < length; parentR++) {
                        compositionalOScore[start][end] += compositionScore[start][parentR][end] *
                                cumlCompositionScore[start][end] * oScoreWParent[start][end][parentR];
                    }
                }
            }
        }

        public void doMuScore() {
            float[][][][] muSplitSpanScoresWParents = preScores.getMuSpanScoreWParent();
            for (int start = 0; start < length; start++) {
                int end = start + 1;
                int split = start;

                float compISplitScore = compositionScore[start][end][split] *
                        cumlCompositionScore[start][split] *
                        cumlCompositionScore[split][end];

                for (int parentL = 0; parentL < start; parentL++) {
                    compositionalMu[start][end][split] += muSplitSpanScoresWParents[start][end][split][parentL] *
                            compISplitScore * compositionScore[parentL][end][start];
                }
                for (int parentR = end; parentR < length; parentR++) {
                    compositionalMu[start][end][split] += muSplitSpanScoresWParents[start][end][split][parentR] *
                            compositionScore[start][parentR][end];
                }

            }

            for (int diff = 2; diff <= length; diff++) {
                for (int start = 0; start + diff <= length; start++) {
                    int end = start + diff;
                    for (int split = start + 1; split < end; split++) {
                        float compISplitScore = compositionScore[start][end][split] *
                                cumlCompositionScore[start][split] *
                                cumlCompositionScore[split][end];

                        for (int parentL = 0; parentL < start; parentL++) {
                            compositionalMu[start][end][split] = muSplitSpanScoresWParents[start][end][split][parentL] *
                                    compISplitScore * compositionScore[parentL][end][start];
                        }
                        for (int parentR = end; parentR < length; parentR++) {
                            compositionalMu[start][end][split] = muSplitSpanScoresWParents[start][end][split][parentR] *
                                    compISplitScore * compositionScore[start][parentR][end];
                        }
                    }
                }
            }
        }

        public CompositionalInsideOutsideScorer(List<Word> sentence) {
            arraySize = 0;
            this.sentence = sentence;
            length = sentence.size();
            preScores = grammar.computeInsideOutsideProb(sentence);
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
            int length = sentence.size();
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


            // Step 2b:
            // Step 2b1: If scorer diff from previous scorer less than tolerance
            // return
            // Step 2b2: Else
            // Step 2b2a: Calculate derivatives

            // Pass compositionalMu to derivate computation functions
            // Let them handle derivative computation
            // Step 2b2b: Do SGD on params
            // Figure out how to do SGD
        } while (true);     // add threshold or tolerance condition
    }

    public void parse(List<Word> sentence) {

    }

    public void nbest(List<List<Word>> decodings) {

    }

    public CompositionalInsideOutsideScorer getScorer(List<Word> sentence) {
        return new CompositionalInsideOutsideScorer(sentence);
    }

}
