package com.kushalarora.compositionalLM.derivatives;

import java.util.HashMap;
import java.util.Map;

import javax.annotation.Nullable;

import com.kushalarora.compositionalLM.lang.StanfordCompositionalInsideOutsideScore;
import com.kushalarora.compositionalLM.optimizer.IIndexedSized;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.google.common.base.Function;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.options.Options;
import com.kushalarora.compositionalLM.utils.Parallelizer;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;


@Slf4j
public class dQdXw<T extends IIndexedSized> extends AbstractBaseDerivativeClass<T> implements IDerivative<T>
{
    @Getter
    private int dim;
    private int V;
    private int length;
    private Options op;
    private Parallelizer parallelizer;
    private static INDArray zLeaf;


    @Getter
    private Map<Integer, INDArray> indexToxMap;

    public dQdXw(int dimensions, int vocabSize, T data, Options op) {
        super(new int[] {dimensions, vocabSize}, data);
        indexToxMap = new HashMap<Integer, INDArray>();
        dim = dimensions;
        V = vocabSize;
        length = data.getSize();
        this.op = op;
        parallelizer = new Parallelizer(op, op.grammarOp.maxLength / op.trainOp.blockNum + 1);
    }


    public dQdXw(dQdXw dqdxw, T data, Options op) {
        super(new int[] {dqdxw.dim, dqdxw.V}, data);
        indexToxMap = dqdxw.indexToxMap;
        dim = dqdxw.dim;
        V = dqdxw.V;
        length = data.getSize();
        this.op = op;
        parallelizer = new Parallelizer(op, op.grammarOp.maxLength / op.trainOp.blockNum + 1);
    }

    private dQdXw(Map<Integer, INDArray> indexToxMap, dQdXw dqdxw, T data, Options op) {
        super(new int[] {dqdxw.dim, dqdxw.V}, data);
        dim = dqdxw.dim;
        V = dqdxw.V;
        length = data.getSize();
        this.indexToxMap = indexToxMap;
        this.op = op;
        parallelizer = new Parallelizer(op, op.grammarOp.maxLength / op.trainOp.blockNum + 1);
    }

    public static INDArray dEdXwUnary(INDArray word, Model model) {

        int dim = model.getDimensions();

        // g'(s)
        double dE = model.energyWordDerivative(word);

        // dE/dXw =  g'(s) X u^T.dot(I_{dXd})
        INDArray dEdXw = model.linearWord( Nd4j.eye(dim)).mul(dE).transpose();

        int[] dEdXwShape = dEdXw.shape();
        if (dEdXwShape[0] != dim ||
                dEdXwShape[1] != 1) {
            throw new RuntimeException("dEdXw unary was expected to be a matrix of shape dim X 1");
        }
        return dEdXw;
    }

    public static INDArray  dEdXwBinary(INDArray dXdXwParent, INDArray dXdXwChild1, INDArray dXdXwChild2,
                                INDArray parent, INDArray child1, INDArray child2,
                                Model model) {

        // g'(s)
        double dE = model.energyCompDerivative(
                parent,
                child1,
                child2);

        // dE/dXw = g'(s) X {u^T.dot(dX_p/dXw) +
        //              h1^T.dot(dX_c1/dXw) +
        //              h2^T.dot(dX_c2/dXw)
        INDArray dEdXw = model.linearComposition(
                        dXdXwParent,
                        dXdXwChild1,
                        dXdXwChild2)
                        .mul(dE)
                        .transpose();

         int[] dEdXwShape = dEdXw.shape();
        if (dEdXwShape[0] != model.getDimensions() ||
                dEdXwShape[1] != 1) {
            throw new RuntimeException("dEdXw binary was expected to be a matrix of shape dim X 1");
        }

        return dEdXw;
    }

    public void calcDerivative(final Model model, final StanfordCompositionalInsideOutsideScore scorer) {

        // Save indexes
        final int[] indexes = new int[length];
        for (int i = 0; i < length; i++) {
            indexes[i] = data.get(i).getIndex();
        }

        final INDArray[][][] compositionMatrix = scorer.getCompositionMatrix();
        final double[][][] compositionalMu = scorer.getCompMuScores();
        final INDArray[][] phraseMatrix = scorer.getPhraseMatrix();
        final double[][] compositionalIScore = scorer.getCompIScores();


        Function<Integer, Void> func = new Function<Integer, Void>() {
            @Nullable
            public Void apply(Integer i) {
                INDArray dQdXw_i = Nd4j.zeros(dim, 1);

                // handle leaf node
                INDArray lineardEdXi_s = dEdXwUnary(phraseMatrix[i][i+1], model);

               dQdXw_i
                   .addi(
                       lineardEdXi_s
	                       .subi(ZLeaf_dEdXw(model, dim, V))
	                       .muli(compositionalMu[i][i + 1][i]));

                final INDArray[][] dxdxwArr =
                    new dXdXwi(dim, V, data, op, i)
                        .calcDerivative(model, scorer);

                // handle the composition case
                for (int diff = 2; diff <= length; diff++) {
                    for (int start = 0; start + diff <= length; start++) {
                        int end = start + diff;

                        INDArray[] lineardEdXi = new INDArray[length];
                        for (int split = start + 1; split < end; split++) {
                            lineardEdXi_s = dEdXwBinary(
                                                dxdxwArr[start][end],
                                                dxdxwArr[start][split],
                                                dxdxwArr[split][end],
                                                compositionMatrix[start][end][split],
                                                phraseMatrix[start][split],
                                                phraseMatrix[split][end],
                                                model);

                            lineardEdXi[split] = lineardEdXi_s;

                            dQdXw_i.addi(lineardEdXi_s
                                .muli(compositionalMu[start][end][split]));
                        }

                        double compMuSum = 0;
                        for (int sp = start + 1; sp < end; sp++) {
                            compMuSum += compositionalMu[start][end][sp];
                        }

                        dQdXw_i.subi(model
                                .Expectedl(start, end, lineardEdXi,
                                        compositionMatrix[start][end],
                                        phraseMatrix, compMuSum,
	                                    new int[]{dim, 1}));
                    }
                }

                if (compositionalIScore[0][length] == 0) {
                    throw new RuntimeException("Z is zero for sentence " + data);
                }

	            dQdXw_i.divi(compositionalIScore[0][length]);

	            if (containsNanOrInf(dQdXw_i)) {
                    log.error("dQdXw contains Nan Or Inf for index: {} data {}::{}. Norm::{}",
                            i, data.getIndex(), data.getSize(), Nd4j.norm2(dQdXw_i));
                    dQdXw_i = Nd4j.zeros(dim);
                }

                indexToxMap.put(indexes[i], clampDerivativeIfNeeded(dQdXw_i));
                return null;
            }
        };

        if (op.trainOp.modelParallel) {
            parallelizer.parallelizer(0, length, func);
        } else {
            for (int i = 0; i < length; i++) {
                func.apply(i);
            }
        }
    }

    public void clear() {
       indexToxMap.clear();
    }

    public void add(IDerivative other) {
        dQdXw odqdxw = (dQdXw)other;

        // for all the values in argument
        for (Map.Entry<Integer,INDArray> entry : ((Map<Integer, INDArray>)(odqdxw.indexToxMap)).entrySet()) {
            INDArray value = entry.getValue();
            Integer key = entry.getKey();
            if (indexToxMap.containsKey(key)) {
                // if key is present in me, add their value to me and store in new map
                indexToxMap.put(key, value.add(indexToxMap.get(key)));
            } else {
                // store them in new map
                indexToxMap.put(key, value);
            }
        }
    }

    public void mul(double learningRate) {
        for (Map.Entry<Integer, INDArray> entry : indexToxMap.entrySet()) {
            indexToxMap.put(entry.getKey(), entry.getValue().mul(learningRate));
        }
    }

    public boolean containsNanOrInf() {
        for (INDArray value : indexToxMap.values()) {
            if (containsNanOrInf(value)) {
                return true;
            }
        }
        return false;
    }

    public IDerivative adaGrad(IDerivative gradient) {
        dQdXw other = (dQdXw)gradient;

        INDArray dqdxw = Nd4j.zeros(dim, V);
        for (Map.Entry<Integer, INDArray> entry : ((Map<Integer, INDArray>)other.indexToxMap).entrySet()) {
            dqdxw.putColumn(entry.getKey(), entry.getValue());
        }

        dqdxw = adaGrad.getGradient(dqdxw, 100);

        Map<Integer, INDArray> newIndexToxMap = new HashMap<Integer, INDArray>();
        for (Map.Entry<Integer, INDArray> entry : ((Map<Integer, INDArray>)other.indexToxMap).entrySet()) {
            Integer key = entry.getKey();
            INDArray value = dqdxw.getColumn(key);
            newIndexToxMap.put(key, value);
        }
        return new dQdXw(newIndexToxMap, this, data, op);
    }

    public double norm()
    {
        double norm = 0;
        for (Map.Entry<Integer, INDArray> entry : indexToxMap.entrySet()) {
            norm += Nd4j.norm2(entry.getValue()).getDouble(0);
        }
        return norm;
    }

    public static INDArray ZLeaf_dEdXw(final Model model, final int dimensions, int vocabSize) {
        if (zLeaf == null) {
            log.info("Calculating ZLeaf_dEdXw");
            zLeaf = model.ExpectedV(new Function<INDArray, INDArray>() {
                @Nullable
                public INDArray apply(@Nullable INDArray indArray) {
                    return dEdXwUnary(indArray, model);
                }
            }, new int[]{dimensions, 1});
        }

        return zLeaf;
    }

    public static void cleanZLeaf() {
        log.info("Cleaning up ZLeaf_dEdXw");
        zLeaf = null;
    }
}
