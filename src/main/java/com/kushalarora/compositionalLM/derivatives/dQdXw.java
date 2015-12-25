package com.kushalarora.compositionalLM.derivatives;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.annotation.Nullable;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.google.common.base.Function;
import com.kushalarora.compositionalLM.model.CompositionalInsideOutsideScore;
import com.kushalarora.compositionalLM.model.Model;
import com.kushalarora.compositionalLM.optimizer.IIndexed;
import com.kushalarora.compositionalLM.options.Options;
import com.kushalarora.compositionalLM.utils.Parallelizer;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;

/**
 * Created by karora on 6/21/15.
 */
@Slf4j
public class dQdXw<T extends List<? extends IIndexed>> extends AbstractBaseDerivativeClass implements IDerivative<T>
{
    @Getter
    private int dim;
    private int V;
    private T data;
    private int length;
    private Options op;
    private Parallelizer parallelizer;

    @Getter
    private Map<Integer, INDArray> indexToxMap;

    public dQdXw(int dimensions, int vocabSize, T data, Options op) {
        super(new int[] {dimensions, vocabSize});
        indexToxMap = new HashMap<Integer, INDArray>();
        dim = dimensions;
        V = vocabSize;
        this.data = data;
        length = data.size();
        this.op = op;
        parallelizer = new Parallelizer(op, op.grammarOp.maxLength / op.trainOp.blockNum + 1);
    }


    public dQdXw(dQdXw dqdxw, T data, Options op) {
        super(new int[] {dqdxw.dim, dqdxw.V});
        indexToxMap = dqdxw.indexToxMap;
        dim = dqdxw.dim;
        V = dqdxw.V;
        this.data = data;
        length = data.size();
        this.op = op;
        parallelizer = new Parallelizer(op, op.grammarOp.maxLength / op.trainOp.blockNum + 1);
    }

    private dQdXw(Map<Integer, INDArray> indexToxMap, dQdXw dqdxw, T data, Options op) {
        super(new int[] {dqdxw.dim, dqdxw.V});
        dim = dqdxw.dim;
        V = dqdxw.V;
        this.data = data;
        length = data.size();
        this.indexToxMap = indexToxMap;
        this.op = op;
        parallelizer = new Parallelizer(op, op.grammarOp.maxLength / op.trainOp.blockNum + 1);
    }

    public void calcDerivative(final Model model, final CompositionalInsideOutsideScore scorer) {

        // Save indexes
        final int[] indexes = new int[length];
        for (int i = 0; i < length; i++) {
            indexes[i] = data.get(i).getIndex();
        }

        final INDArray[][][][] dxdxwArr = new dXdXw(dim, V, data, op).calcDerivative(model, scorer);
        final INDArray[][][] compositionMatrix = scorer.getCompositionMatrix();
        final double[][][] compositionalMu = scorer.getMuScore();
        final INDArray[][] phraseMatrix = scorer.getPhraseMatrix();
        final double[][] compositionalIScore = scorer.getInsideSpanProb();


        final INDArray dcdc = Nd4j.eye(dim);
        Function<Integer, Void> func = new Function<Integer, Void>() {
            @Nullable
            public Void apply(Integer i) {
                INDArray dQdXw_i = Nd4j.zeros(dim);

                // handle leaf node
                INDArray vector = phraseMatrix[i][i + 1];
                double dE = model.energyDerivative(vector);

                // diff wrt to self returns eye
                INDArray udXdXwArr =
                        model.getParams().getU()
                             .mmul(dcdc);

                int[] udXdXwShape = udXdXwArr.shape();
                if (udXdXwShape.length != 1 ||
                        udXdXwShape[0] != dim) {
                    throw new RuntimeException("udXdXwArr was expected to be a matrix of shape dim X 1 " + udXdXwShape.toString());
                }

                dQdXw_i = dQdXw_i.add(udXdXwArr
                        .muli(compositionalMu[i][i + 1][i]))
                        .muli(dE);

                // handle the composition case
                for (int diff = 2; diff <= length; diff++) {
                    for (int start = 0; start + diff <= length; start++) {
                        int end = start + diff;
                        for (int split = start + 1; split < end; split++) {
                            dE = model.energyDerivative(compositionMatrix[start][end][split],
                                    phraseMatrix[start][split], phraseMatrix[split][end]);

                            udXdXwArr =
                                    model
                                            .getParams()
                                            .getU()
                                            .mmul(dxdxwArr[i][start][end][split]);


                            udXdXwShape = udXdXwArr.shape();
                            if (udXdXwShape.length != 1 ||
                                    udXdXwShape[0] != dim) {
                                throw new RuntimeException("udXdXwArr was expected to be a matrix of shape dim X 1");
                            }

                            dQdXw_i = dQdXw_i.add(udXdXwArr.muli(compositionalMu[start][end][split]))
                                    .muli(dE);
                        }
                    }
                }

                dQdXw_i = dQdXw_i.div(compositionalIScore[0][length]);
                if (containsNanOrInf(dQdXw_i)) {
                    log.error("dQdXw contains Nan Or Inf for index: {} data: {}", i, data);
                    dQdXw_i = Nd4j.zeros(dim);
                }

                indexToxMap.put(indexes[i], dQdXw_i);
                return null;
            }
        };

        if (op.trainOp.parallel) {
            parallelizer.parallelizer(0, length, func);
        } else {
            for (int i = 0; i < length; i++) {
                func.apply(i);
            }
        }

        if (compositionalIScore[0][length] == 0) {
            throw new RuntimeException("Z is zero for sentence " + data);
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

        dqdxw = adaGrad.getGradient(dqdxw);

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
            norm += Nd4j.norm2(entry.getValue()).sum(Integer.MAX_VALUE).getFloat(0);
        }
        return norm;
    }
}
