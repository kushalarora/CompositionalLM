package com.kushalarora.compositionalLM.lang;

import edu.stanford.nlp.parser.lexparser.BinaryRule;
import edu.stanford.nlp.parser.lexparser.Lexicon;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.ujmp.core.SparseMatrix;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

@Slf4j
public class StanfordCompositionalInsideOutsideScore extends AbstractInsideOutsideScore {
    // Averaged representation of phrases in sentence
    protected transient INDArray[][] phraseMatrix;

    // composition matrix for all possible phrases
    // contains multiple representation for each
    // phrase originating from different split
    protected transient INDArray[][][] compositionMatrix;

    // extended mu to included compositional score
    protected transient double[][][] compositionalMu;

    // extended inside score with compositional score
    protected transient double[][] compIScore;

    protected transient double[][][] compISplitScore;

    // inside scores
    // start idx, end idx, state -> logProb (ragged; null for end <= start)
    protected transient SparseMatrix iScore;

    // outside scores
    // start idx, end idx, state -> logProb
    protected transient SparseMatrix oScore;

    protected transient SparseMatrix iSplitSpanStateScore;

    protected transient SparseMatrix muScore;

    @Getter
    protected Sentence sentence;

    @Getter
    protected int length;

    @Getter
    protected int numStates;

    Set<BinaryRule> binaryRuleSet;




    public StanfordCompositionalInsideOutsideScore(Sentence sentence, int dimensions, int numStates) {
        super(sentence.size() + 1, numStates);
        this.sentence = new Sentence(sentence.getIndex());
        this.sentence.addAll(sentence);
        this.sentence.add(new Word(Lexicon.BOUNDARY, sentence.size()));
        length = this.sentence.size();
        log.info("Creating Compositional matrices for length {}: {}", length, sentence.getIndex());
        int dim = dimensions;

        phraseMatrix = new INDArray[length][length + 1];

        compositionMatrix = new INDArray[length][length + 1][];
        for (int start = 0; start < length; start++) {
            for (int end = start + 1; end <= length; end++) {
                phraseMatrix[start][end] = Nd4j.zeros(dim, 1);
                compositionMatrix[start][end] = new INDArray[length];
                for (int split = start; split < end; split++) {
                    compositionMatrix[start][end][split] = Nd4j.zeros(dim, 1);
                }
            }
        }

        compositionalMu = new double[length][length + 1][];
        for (int start = 0; start < length; start++) {
            for (int end = start + 1; end <= length; end++) {
                compositionalMu[start][end] = new double[length];
            }
        }

        compIScore = new double[length][length + 1];

        compISplitScore = new double[length][length + 1][];
        for (int start = 0; start < length; start++) {
            for (int end = start + 1; end <= length; end++) {
                compISplitScore[start][end] = new double[length];
            }
        }

        iScore = SparseMatrix.Factory.zeros(length, length + 1, numStates);

        oScore = SparseMatrix.Factory.zeros(length, length + 1, numStates);

        iSplitSpanStateScore = SparseMatrix.Factory.zeros(length, length + 1, length, numStates);

        binaryRuleSet = Collections.synchronizedSet(new HashSet<BinaryRule>());
    }

    public double[][] getCompIScores() {
        return compIScore;
    }

    public double[][][] getCompISplitScore() {
        return compISplitScore;
    }

    public double[][][] getCompMuScores() {
        return compositionalMu;
    }

    public INDArray[][][] getCompositionMatrix() {
        return compositionMatrix;
    }

    public INDArray[][] getPhraseMatrix() {
        return phraseMatrix;
    }

    public SparseMatrix getInsideScores() {
        return iScore;
    }

    public SparseMatrix getOutsideScores() {
        return oScore;
    }

    public SparseMatrix getMuScores() {
        return muScore;
    }

    public double getSentenceScore() {
        double score = compIScore[0][length];
        if (score == 0) {
            log.error("Score is 0 for sentence : {}", sentence);
            return Double.NEGATIVE_INFINITY;
        }
        if (Double.isInfinite(compIScore[0][length])) {
            log.error("Score is Nan or Inf for sentence : {}", sentence);
            return Double.POSITIVE_INFINITY;
        }

        if (compIScore[0][length] < 0) {
            log.error("Score is negative for sentence: {}", sentence);
            return Double.NEGATIVE_INFINITY;
        }
        return Math.log(compIScore[0][length]);
    }

    public double getScore(SparseMatrix matrix, long... indexes) {
        return matrix.getAsDouble(indexes);
    }

    protected synchronized void setScore(SparseMatrix matrix, double value, long... indexes) {
        matrix.setAsDouble(value, indexes);
    }

    protected synchronized void addToScore(SparseMatrix matrix, double value, long... indexes) {
        setScore(matrix, value + getScore(matrix, indexes), indexes);
    }

    public void postProcess() {

        Sentence newSentence = new Sentence(sentence.getIndex());
        for (int i = 0; i < sentence.getSize() - 1; i++) {
            newSentence.add(sentence.get(i));
        }
        this.sentence = newSentence;
        this.length = sentence.getSize();
        iScore = null;
        oScore = null;
        iSplitSpanStateScore = null;
        muScore = null;
    }

}
