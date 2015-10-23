import sys
import os
from deform_test import disform_sentences
import subprocess
import json
import csv
from math import log10

MODELS = [
        "3gram",
        "5gram",
        "3gram-kn",
        "5gram-kn",
        "3gram-gt",
        "5gram-gt",
        "5max-ent",
        "3max-ent",
        "rnn-100-100",
        "rnn-200-100",
        "rnn-50-100",
        "rnn-20-100",
        ]
ngramModelsToTrainCmd = {
        "3gram": "ngram-count -order 3 -text %s -lm %s -unk -addsmooth 1 -interpolate",
        "5gram": "ngram-count -order 5 -text %s -lm %s -unk -addsmooth 1 -interpolate",
        "3gram-kn": "ngram-count -order 3 -text %s -lm %s -unk -kndiscount -interpolate -gt3min 1 -gt4min 1",
        "5gram-kn":"ngram-count -order 5 -text %s -lm %s -unk -kndiscount -interpolate -gt3min 1 -gt4min 1",
        "3gram-gt": "ngram-count -order 3 -text %s -lm %s -unk -interpolate -gt3min 1 -gt4min 1",
        "5gram-gt": "ngram-count -order 5 -text %s -lm %s -unk -interpolate -gt3min 1 -gt4min 1",
        "5max-ent": "ngram-count -order 5 -text %s -lm %s -unk  -gt3min 1 -gt4min 1",
        "3max-ent": "ngram-count -order 3 -text %s -lm %s -unk  -gt3min 1 -gt4min 1"
        }
ngramModelToTestCmds = {
        "3gram": "ngram -order 3 -ppl %s -lm %s -unk",
        "5gram": "ngram -order 5 -ppl %s -lm %s -unk",
        "3gram-kn": "ngram -order 3 -ppl %s -lm %s -unk",
        "5gram-kn":"ngram -order 5 -ppl %s -lm %s -unk",
        "3gram-gt": "ngram -order 3 -ppl %s -lm %s -unk",
        "5gram-gt": "ngram -order 5 -ppl %s -lm %s -unk",
        "5max-ent": "ngram -order 5 -ppl %s -lm %s -unk",
        "3max-ent": "ngram -order 3 -ppl %s -lm %s -unk"
        }

DEFORM_LEVELS = [0, 5, 10, 15, 20, 25]

iter = 1

TMP_DIR = "/tmp/"
TRAIN_FILE_PATH="../src/resources/ptb.train.txt"
VALID_FILE_PATH = "../src/resources/ptb.valid.txt"
TEST_FILE_PATH="../src/resources/ptb.test.txt"
TEST_SENT_COUNT = 3761
TEST_VOCAB = 78669

RNN_TRAIN_CMD = "../rnnlm/rnnlm -train %s -valid %s -rnnlm %s -hidden %d -rand-seed 1 -class %d -bptt 4 -bptt-block 10"
RNN_TEST_CMD = "../rnnlm/rnnlm -test %s -rnnlm %s"

RNN_CLASS_HIDDEN = [(100, 100), (200, 100), (50, 100), (20, 100)]

def get_model_filename(model):
    return os.path.join(TMP_DIR, model)

def generate_results():
    deformToModelToScore = {}
    for model in ngramModelsToTrainCmd:
        # train model
        model_filepath = get_model_filename(model)
        if os.path.exists(model_filepath):
            continue
        print "Training Model %s" % model
        cmd = ngramModelsToTrainCmd[model] % (TRAIN_FILE_PATH, model_filepath)
        print subprocess.call(cmd.split(" "))
    for tuple in RNN_CLASS_HIDDEN:
        model_file = get_model_filename("rnn-%d-%d" % (tuple[0], tuple[1]))

        if os.path.exists(model_file):
            continue
        print "Training RNN Model %s" % model_file
        cmd = RNN_TRAIN_CMD % (TRAIN_FILE_PATH, VALID_FILE_PATH, model_file, tuple[0], tuple[1])
        print subprocess.call(cmd.split(" "))

    for deform in DEFORM_LEVELS:
        deformToModelToScore[deform] = {}
        cuml_logp = 0
        print "Deform Level: %d" % deform
        for i in xrange(iter):
            print "Iter: %d" % i
            pn = 100 - 2 * deform
            ps = pt = deform
            output_filename = disform_sentences(TEST_FILE_PATH, pn, ps, pt)
            print output_filename
            for model in ngramModelToTestCmds:

                if model not in deformToModelToScore[deform]:
                    deformToModelToScore[deform][model] = 0.0

                print "Model: %s" % model
                cmd = ngramModelToTestCmds[model] % (output_filename, get_model_filename(model))
                output = subprocess.Popen(cmd.split(" "), stdout=subprocess.PIPE).stdout.read().strip()
                print output
                words = output.split(" ")
                deformToModelToScore[deform][model] += float(words[-5])

            for tuple in RNN_CLASS_HIDDEN:
                model = "rnn-%d-%d" % tuple
                print "Model: %s" % model
                if model not in deformToModelToScore[deform]:
                    deformToModelToScore[deform][model] = 0.0

                cmd = RNN_TEST_CMD % (output_filename, get_model_filename(model))
                output = subprocess.Popen(cmd.split(" "), stdout=subprocess.PIPE).stdout.read().strip()
                print output
                line = output.split("\n\n")[1]
                deformToModelToScore[deform][model] += float(line.split(":")[1])

    for model in MODELS:
        for deform in DEFORM_LEVELS:
            deformToModelToScore[deform][model] /= iter

    outputfile = open("./output.txt", "w")
    outputfile.write("deformToModelToScore: " + json.dumps(deformToModelToScore, sort_keys=True, indent=True, separators=(',', ':')))
    outputfile.write("\n")
    deformToModelContrastiveEntropy = {}

    for deform in DEFORM_LEVELS:
        deformToModelContrastiveEntropy[deform] = {}
        for model in deformToModelToScore[deform]:
            deformToModelContrastiveEntropy[deform][model] = deformToModelToScore[deform][model] - deformToModelToScore[0][model]

    outputfile.write("deformToModelContrastiveEntropy: " + json.dumps(deformToModelContrastiveEntropy, sort_keys=True, indent=True, separators=(',', ':')))

    import pdb;pdb.set_trace()
    deformToModelToPpl = {}
    for deform in DEFORM_LEVELS:
        deformToModelToPpl[deform] = {}
        for model in MODELS:
            deformToModelToPpl[deform][model] = pow(10, -1 * deformToModelToScore[deform][model]/(TEST_VOCAB + TEST_SENT_COUNT))

    outputfile.write("deformToModelToPpl: " + json.dumps(deformToModelToPpl, sort_keys=True, indent=True, separators=(',', ':')))

    deformToModelToContrastivePPl = {}
    for deform in DEFORM_LEVELS:
        deformToModelToContrastivePPl[deform] = {}
        for model in MODELS:
            deformToModelToContrastivePPl[deform][model] = pow(10, -1 * deformToModelContrastiveEntropy[deform][model]/(TEST_VOCAB + TEST_SENT_COUNT))
    outputfile.write("deformToModelToContrastivePPl: " + json.dumps(deformToModelToContrastivePPl, sort_keys=True, indent=True, separators=(',', ':')))
    outputfile.close()

if __name__ == "__main__":
	generate_results()
