package com.kushalarora.compositionalLM.model;

import com.kushalarora.compositionalLM.lang.Sentence;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static java.lang.Math.exp;

/**
 * Created by arorak on 9/15/15.
 */
@Slf4j
public class CompositionalInsideOutsideScore {
    // Averaged representation of phrases in sentence
    protected transient INDArray[][] phraseMatrix;

    // composition matrix for all possible phrases
    // contains multiple representation for each
    // phrase originating from different split
    protected transient INDArray[][][] compositionMatrix;

    // extended mu to included compositional score
    protected transient double[][][] compositionalMu;

    // extended inside score with compositional score
    protected transient double[][] compositionalIScore;

    protected transient double[][][] compositionISplitScore;

    // extended outside score extended with compositional score
    protected transient double[][] compositionalOScore;

    // score for each phrase composition.
    protected transient double[][][] compositionScore;

    // sum of scores all possible composition of a phrase.
    // marginalization over split
    protected transient double[][] cumlCompositionScore;

    @Getter
    protected Sentence sentence;

    protected int length;


    public CompositionalInsideOutsideScore(Sentence sentence, int dimensions) {
        this.sentence = sentence;
        length = sentence.size();
        log.info("Creating Compositional matrices for length {}: {}", length, sentence.getIndex());
        int dim = dimensions;
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


}
