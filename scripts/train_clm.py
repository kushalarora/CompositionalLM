#!/usr/bin/env python
"""Sample script of Compositional Language Model.
"""

import argparse
import codecs
import collections
import random
import re
import time

import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from datasets import build_vocab, index_data


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=400, type=int,
                    help='number of epochs to learn')
parser.add_argument('--unit', '-u', default=30, type=int,
                    help='number of units')
parser.add_argument('--batchsize', '-b', type=int, default=25,
                    help='learning minibatch size')
parser.add_argument('--label', '-l', type=int, default=5,
                    help='number of labels')
parser.add_argument('--epocheval', '-p', type=int, default=5,
                    help='number of epochs per evaluation')
parser.add_argument('--test', dest='test', action='store_true')
parser.set_defaults(test=False)
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

n_epoch = args.epoch       # number of epochs
n_units = args.unit        # number of units per layer
batchsize = args.batchsize      # minibatch size
n_label = args.label         # number of labels
epoch_per_eval = args.epocheval  # number of epochs per evaluation


class SexpParser(object):

    def __init__(self, line):
        self.tokens = re.findall(r'\(|\)|[^\(\) ]+', line)
        self.pos = 0

    def parse(self):
        assert self.pos < len(self.tokens)
        token = self.tokens[self.pos]
        assert token != ')'
        self.pos += 1

        if token == '(':
            children = []
            while True:
                assert self.pos < len(self.tokens)
                if self.tokens[self.pos] == ')':
                    self.pos += 1
                    break
                else:
                    children.append(self.parse())
            return children
        else:
            return token


def convert_tree(vocab, exp):
    assert isinstance(exp, list) and (len(exp) == 2 or len(exp) == 3)

    if len(exp) == 2:
        label, leaf = exp
        if leaf not in vocab:
            vocab[leaf] = len(vocab)
        return {'label': int(label), 'node': vocab[leaf]}
    elif len(exp) == 3:
        label, left, right = exp
        node = (convert_tree(vocab, left), convert_tree(vocab, right))
        return {'label': int(label), 'node': node}


def read_corpus(path, vocab, max_size):
    with codecs.open(path, encoding='utf-8') as f:
        trees = []
        for line in f:
            line = line.strip()
            tree = SexpParser(line).parse()
            trees.append(convert_tree(vocab, tree))
            if max_size and len(trees) >= max_size:
                break

        return trees


class RecursiveNet(chainer.Chain):

    def __init__(self, n_vocab, n_units):
        super(RecursiveNet, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            l=L.Linear(n_units * 2, n_units),
            w=L.Linear(n_units, n_label))

    def leaf(self, x):
        return self.embed(x)

    def node(self, left, right):
        return F.tanh(self.l(F.concat((left, right))))

    def label(self, v):
        return self.w(v)


def traverse(model, node, train=True, evaluate=None, root=True):
    if isinstance(node['node'], int):
        # leaf node
        word = xp.array([node['node']], np.int32)
        loss = 0
        x = chainer.Variable(word, volatile=not train)
        v = model.leaf(x)
    else:
        # internal node
        left_node, right_node = node['node']
        left_loss, left = traverse(
            model, left_node, train=train, evaluate=evaluate, root=False)
        right_loss, right = traverse(
            model, right_node, train=train, evaluate=evaluate, root=False)
        v = model.node(left, right)
        loss = left_loss + right_loss

    y = model.label(v)

    if train:
        label = xp.array([node['label']], np.int32)
        t = chainer.Variable(label, volatile=not train)
        loss += F.softmax_cross_entropy(y, t)

    if evaluate is not None:
        predict = cuda.to_cpu(y.data.argmax(1))
        if predict[0] == node['label']:
            evaluate['correct_node'] += 1
        evaluate['total_node'] += 1

        if root:
            if predict[0] == node['label']:
                evaluate['correct_root'] += 1
            evaluate['total_root'] += 1

    return loss, v


def evaluate(model, test_trees):
    m = model.copy()
    m.volatile = True
    result = collections.defaultdict(lambda: 0)
    for tree in test_trees:
        traverse(m, tree, train=False, evaluate=result)

    acc_node = 100.0 * result['correct_node'] / result['total_node']
    acc_root = 100.0 * result['correct_root'] / result['total_root']
    print(' Node accuracy: {0:.2f} %% ({1:,d}/{2:,d})'.format(
        acc_node, result['correct_node'], result['total_node']))
    print(' Root accuracy: {0:.2f} %% ({1:,d}/{2:,d})'.format(
        acc_root, result['correct_root'], result['total_root']))

vocab = {}
if args.test:
    max_size = 10
else:
    max_size = None
train_trees = read_corpus('trees/train.txt', vocab, max_size)
test_trees = read_corpus('trees/test.txt', vocab, max_size)
develop_trees = read_corpus('trees/dev.txt', vocab, max_size)

model = RecursiveNet(len(vocab), n_units)

if args.gpu >= 0:
    model.to_gpu()

# Setup optimizer
optimizer = optimizers.AdaGrad(lr=0.1)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

accum_loss = 0
count = 0
start_at = time.time()
cur_at = start_at
for epoch in range(n_epoch):
    print('Epoch: {0:d}'.format(epoch))
    total_loss = 0
    cur_at = time.time()
    random.shuffle(sentences)
    for tree in train_trees:
        loss, v = traverse(model, tree, train=True)
        accum_loss += loss
        count += 1

        if count >= batchsize:
            model.cleargrads()
            accum_loss.backward()
            optimizer.update()
            total_loss += float(accum_loss.data)

            accum_loss = 0
            count = 0

    print('loss: {:.2f}'.format(total_loss))

    now = time.time()
    throughput = float(len(train_trees)) / (now - cur_at)
    print('{:.2f} iters/sec, {:.2f} sec'.format(throughput, now - cur_at))
    print()

    if (epoch + 1) % epoch_per_eval == 0:
        print('Train data evaluation:')
        evaluate(model, train_trees)
        print('Develop data evaluation:')
        evaluate(model, develop_trees)
        print('')

print('Test evaluateion')
evaluate(model, test_trees)

TRAIN_FILE_PATH="../src/resources/ptb.train.txt"
VALID_FILE_PATH = "../src/resources/ptb.valid.txt"
TEST_FILE_PATH="../src/resources/ptb.test.txt"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=20,
                        help='Number of examples in each mini-batch')

    parser.add_argument('--bproplen', '-l', type=int, default=35,
                        help='Number of words in each mini-batch '
                             '(= length of truncated BPTT)')

    parser.add_argument('--epoch', '-e', type=int, default=39,
                        help='Number of sweeps over the dataset to train')

    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')

    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')

    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')

    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')

    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')

    parser.set_defaults(test=False)

    parser.add_argument('--unit', '-u', type=int, default=650,
                        help='Number of LSTM units in each layer')

    args = parser.parse_args()

    # Load the Penn Tree Bank long word sequence dataset
    train, vocab, vocab_index = build_vocab(TRAIN_FILE_PATH)
    valid = index_data(VALID_FILE_PATH)
    test = index_data(TEST_FILE_PATH)

    n_vocab = max(train) + 1  # train is just an array of integers
    print('#vocab =', n_vocab)

    import pdb;pdb.set_trace()
    if args.test:
        train = train[:100]
        val = val[:100]
        test = test[:100]

    train_iter = ParallelSequentialIterator(train, args.batchsize)
    val_iter = ParallelSequentialIterator(val, 1, repeat=False)
    test_iter = ParallelSequentialIterator(test, 1, repeat=False)

    # Prepare an RNNLM model
    rnn = RNNForLM(n_vocab, args.unit)
    model = L.Classifier(rnn)
    model.compute_accuracy = False  # we only want the perplexity
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # make the GPU current
        model.to_gpu()

    # Set up an optimizer
    optimizer = chainer.optimizers.SGD(lr=1.0)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

    # Set up a trainer
    updater = BPTTUpdater(train_iter, optimizer, args.bproplen, args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    eval_model = model.copy()  # Model with shared params and distinct states
    eval_rnn = eval_model.predictor
    eval_rnn.train = False
    trainer.extend(extensions.Evaluator(
        val_iter, eval_model, device=args.gpu,
        # Reset the RNN state at the beginning of each evaluation
        eval_hook=lambda _: eval_rnn.reset_state()))

    interval = 10 if args.test else 500
    trainer.extend(extensions.LogReport(postprocess=compute_perplexity,
                                        trigger=(interval, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'perplexity', 'val_perplexity']
    ), trigger=(interval, 'iteration'))
    trainer.extend(extensions.ProgressBar(
        update_interval=1 if args.test else 10))
    trainer.extend(extensions.snapshot())
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'))
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

    # Evaluate the final model
    print('test')
    eval_rnn.reset_state()
    evaluator = extensions.Evaluator(test_iter, eval_model, device=args.gpu)
    result = evaluator()
    print('test perplexity:', np.exp(float(result['main/loss'])))


if __name__ == '__main__':
    main()
