import sys
import os
from deform_test import disform_sentences
import subprocess
import json

MODELS = [
        "3gram",
        "5gram",
        "3gram-kn",
        "5gram-kn",
        "3gram-gt",
        "5gram-gt",
        ]
modelToTrainCmds = {
        "3gram": "ngram-count -order 3 -text %s -lm /tmp/3gramlm -unk -addsmooth 1 -interpolate",
        "5gram": "ngram-count -order 5 -text %s -lm /tmp/5gram -unk -addsmooth 1 -interpolate",
        "3gram-kn": "ngram-count -order 3 -text %s -lm /tmp/3gram-kn -unk -kndiscount -interpolate",
        "5gram-kn":"ngram-count -order 5 -text %s -lm /tmp/5gram-kn -unk -kndiscount -interpolate",
        "3gram-gt": "ngram-count -order 3 -text %s -lm /tmp/3gram-gt -unk -interpolate",
        "5gram-gt": "ngram-count -order 5 -text %s -lm /tmp/5gram-gt -unk -interpolate",
        }
modelToTestCmds = {
        "3gram": "ngram -order 3 -ppl %s -lm /tmp/3gramlm -unk",
        "5gram": "ngram -order 5 -ppl %s -lm /tmp/5gram -unk",
        "3gram-kn": "ngram -order 3 -ppl %s -lm /tmp/3gram-kn -unk",
        "5gram-kn":"ngram -order 5 -ppl %s -lm /tmp/5gram-kn -unk",
        "3gram-gt": "ngram -order 3 -ppl %s -lm /tmp/3gram-gt -unk",
        "5gram-gt": "ngram -order 5 -ppl %s -lm /tmp/5gram-gt -unk"
        }

DEFORM_LEVELS = [0, 5, 10, 15, 20, 25]

iter = 10

TMP_DIR = "/tmp/"
TRAIN_FILE_PATH="../src/resources/ptb.train.txt"
VALID_FILE_PATH = "src/resources/ptb.valid.txt"
TEST_FILE_PATH="../src/resources/ptb.test.txt"

RNN_TRAIN_CMD = "../rnnlm/rnnlm -train %s -valid %s -rnnlm /tmp/rnnlm-100 -hidden %d -rand-seed 1 -class 100 -bptt 4 -bptt-block 10",

def generate_results():
    modeToDeformToScorelMap = {}
    for model in MODELS:
        # train model
        print "Training Model %s" % model
        cmd = modelToTrainCmds[model] % TRAIN_FILE_PATH
        modeToDeformToScorelMap[model] = {}
        print subprocess.call(cmd.split(" "))

    for deform in DEFORM_LEVELS:
        cuml_logp = 0
        print "Deform Level: %d" % deform
        for i in xrange(iter):
            print "Iter: %d" % i
            pn = 100 - 2 * deform
            ps = pt = deform
            output_filename = disform_sentences(TEST_FILE_PATH, pn, ps, pt)
            print output_filename
            for model in MODELS:
                print "Model: %s" % model
                cmd = modelToTestCmds[model] % output_filename
                output = subprocess.Popen(cmd.split(" "), stdout=subprocess.PIPE).stdout.read().strip()
                print output
                words = output.split(" ")
                cuml_logp += float(words[-5])
                avg_logp = cuml_logp/iter
            print "Avg log P: %d" % avg_logp
            modeToDeformToScorelMap[model][deform] = avg_logp

    outputfile = open("./output.txt", "w")
    outputfile.write(json.dumps(modeToDeformToScorelMap, sort_keys=True, indent=True, separators=(',', ':')))
    outputfile.write("\n")
    import pdb;pdb.set_trace()
    modelToDeformContrastiveScore = {}
    for model, deformMap in modeToDeformToScorelMap.iteritems():
        modelToDeformContrastiveScore[model] = {}
        deformToScoreMap = modeToDeformToScorelMap[model]
        for deform in DEFORM_LEVELS[1:]:
            modelToDeformContrastiveScore[model][deform] = deformToScoreMap[deform] - deformToScoreMap[0]


    outputfile.write(json.dumps(modelToDeformContrastiveScore, sort_keys=True, indent=True, separators=(',', ':')))

if __name__ == "__main__":
	generate_results()
