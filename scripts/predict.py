import tensorflow as tf
import numpy as np
from functools import partial
import json, sys, os
from model import DNAModel
import itertools
import model_utils

import argparse
parser = argparse.ArgumentParser(description='Predict any sequence.')
parser.add_argument("--feature", dest = "feature", help = "")
parser.add_argument("--seq", dest = "seq", help = "")
parser.add_argument("--file",dest = "file", help = "If no seq is provided, use file to read in all seqs.")
parser.add_argument("--output", dest = "output", default = "prediction.txt")
parser.add_argument("--steps", dest = "steps", default = 10, type = int)
parser.add_argument("--batch_size", dest = "batch_size", default = 1, type = int)
parser.add_argument("--mp_layers", dest = "mp_layers", default = 1, type = int)
parser.add_argument("--filters", dest = "filters", default = 64, type = int)
parser.add_argument("--weight_decay", dest = "weight_decay", default = 0.0, type = float)
parser.add_argument("--multiply", dest = "multiply", action = "store_true", help = "when selected, change the model to use 'multiply' rather than 'add'")
parser.add_argument("--add", dest = "add", action = "store_true", help = "When select multiply, choose if you want to add another flavor of 'add'.")
parser.add_argument("--alllayers", dest = "constraints", action = "store_true")
parser.add_argument("--self_layer", dest = "if_self_layer", action = "store_true")
parser.add_argument("--padding", dest = "padding", action = "store_true", help = "Activate this option will add 2 Ns to the terminal of sequences. For example, ACGTA will become NNACGTANN.")
parser.add_argument("--phychem_encoding", dest = "if_phychem", action = "store_true")
parser.add_argument("--5mc", dest = "if_5mc", action = "store_true")
parser.add_argument("--nobn", dest = "if_nobn", action = "store_true")
parser.add_argument("--nogru", dest = "if_nogru", action = "store_true", help = "If set, replace gru cell with a sigmoid function.")
args = parser.parse_args()

x_axis_features = set(['Shift', 'Tilt', 'Shear', 'Buckle'])
intrabase_features = {"Shear", "Stretch", "Stagger", "Buckle", "ProT", "Opening", "MGW", "EP", "Shear-FL", "Stretch-FL", "Stagger-FL", "Buckle-FL", "ProT-FL", "Opening-FL", "MGW-FL",  "EP-FL"}
interbase_features = {"Shift", "Slide", "Rise", "Tilt", "Roll", "HelT", "Shift-FL", "Slide-FL", "Rise-FL", "Tilt-FL", "Roll-FL", "HelT-FL"}
feature = args.feature
featuremodel = feature
if len(feature.split("_")) > 1:
    feature = feature.split("_")[0]
basestepfeatures = 1 if feature in interbase_features else 0
basefeatures = 1 if feature in intrabase_features else 0
seq = args.seq
filename = args.file
outputfile = args.output
if_nobn = args.if_nobn
if_nogru = args.if_nogru
filters = args.filters
if_self_layer = args.if_self_layer
padded = args.padding
if padded:
    exclude_bp = 2
weight_decay = args.weight_decay
multiply = args.multiply
ifadd = args.add
if multiply and ifadd:
    multiply = "add"
ifconstraints = args.constraints
batch_size = args.batch_size

with open("params.txt") as json_file:
    minmax_params = json.load(json_file)

#import tensorflow_probability as tfp

pairs, diPairs = model_utils.getBasesMapping(args.if_phychem, args.if_5mc)

def oneHot(seq, padding = False):
    #if padding:
    #    seq = "N" + seq + "N"
    return np.array(list(map(lambda x: pairs[x], seq)))
def revSeq(seq):
    revpairs = {"A":"T", "T":"A", "C":"G", "G":"C", "N": "N", "M": "g", "g": "M"}
    return "".join(list(map(lambda x: revpairs[x], seq))[::-1])
def oneHotDi(seq, padding = False):
    #if padding:
    #    seq = "N" + seq + "N"
    return np.array(list(map(lambda i: diPairs[(seq[i], seq[i+1])], range(len(seq)-1))))
def oneHotDi2(seq, padding = False):
    #if padding:
    #    seq = "N" + seq + "N"
    return np.array(list(map(lambda i: pairs[seq[i]] + pairs[seq[i+1]], range(len(seq)-1))))


def preprocess(x):
    #x is k * 4
    num_linkages = x.shape[0]
    linkages = tf.range(num_linkages - 1, dtype = tf.int64)
    linkages = tf.reshape(tf.repeat(linkages, 4), (-1, 4)) + tf.pad(tf.ones((num_linkages - 1,2), dtype = tf.int64), ((0,0), (1,1)))
    return x, tf.reshape(linkages, (-1, 2))

def preprocess_with_selfloop(x):
    #x is k * 4
    num_linkages = x.shape[0]
    linkages = tf.range(num_linkages - 1, dtype = tf.int64)
    linkages = tf.reshape(tf.repeat(linkages, 4), (-1, 4)) + tf.pad(tf.ones((num_linkages - 1,2), dtype = tf.int64), ((0,0), (1,1)))
    linkages = tf.reshape(linkages, (-1, 2))
    linkages = tf.concat([tf.zeros((1, 2), dtype = tf.int64), linkages, tf.ones((1, 2), dtype = tf.int64) * (num_linkages - 1)], axis = 0)
    linkages = tf.reshape(linkages, (-1, 4))
    linkagesPrev = linkages[:, :2]
    linkagesNext = linkages[:, 2:]
    return x, (linkagesPrev, linkagesNext)
    #return x, linkages

def prebatch_with_selfloop(inputs):
    #inputs is ragged tensor, shape of (batch size, None, 4)
    mergedInput = inputs.merge_dims(outer_axis = 0, inner_axis = 1)
    batch_size = inputs.shape[0]
    kmers = inputs.row_lengths() # [k1, k2, k3, ..., kb]
    num_linkages = tf.reduce_sum(kmers) - batch_size
    linkages = tf.range(num_linkages, dtype = tf.int64) #raw 
    linkages = tf.reshape(tf.repeat(linkages, 4), (-1, 4)) + tf.pad(tf.ones((num_linkages,2), dtype = tf.int64), ((0,0), (1,1))) #add all connections
    linkages = linkages + tf.reshape(tf.repeat(tf.repeat(tf.range(batch_size, dtype = tf.int64), (kmers - 1)), 4), (-1, 4)) #add skip to seperate DNAs

    linkages = tf.reshape(linkages, [-1])
    linkages = tf.RaggedTensor.from_row_lengths(linkages, (kmers - 1) * 4)
    pad = tf.cumsum(kmers)
    pad1 = tf.reshape(tf.repeat(pad - kmers, 2), (-1, 2))
    pad2 = tf.reshape(tf.repeat(pad - 1, 2), (-1, 2))
    linkages = tf.concat([pad1, linkages, pad2], axis = 1)
    linkages = linkages.merge_dims(0,1)
    #linkages = tf.reshape(linkages, (-1, 2)) #reshape to linkage
    linkages = tf.reshape(linkages, (-1, 4))
    linkagesPrev = linkages[:, :2]
    linkagesNext = linkages[:, 2:]
    return (mergedInput, (linkagesPrev, linkagesNext)), kmers

def preprocess_with_bonds(x):
    num_linkages = x.shape[0] - 1
    #bond_features = tf.zeros((num_linkages, 1), dtype = tf.float32)
    linkages = tf.range(num_linkages, dtype = tf.int64)
    linkages = tf.reshape(tf.repeat(linkages, 4), (-1, 4)) 
    #bond_linkages = linkages + tf.pad(tf.ones((num_linkages,1), dtype = tf.int64), ((0,0), (3,0))) #add connections
    #bond_linkages = tf.reshape(bond_linkages, (-1, 2))
    linkages = linkages + tf.pad(tf.ones((num_linkages,2), dtype = tf.int64), ((0,0), (1,1)))
    return x, tf.reshape(linkages, (-1, 2))

def rescale(predictions, minmax_params):
    method = minmax_params[feature]["method"]
    if method == "minmax":
        rescaled_predictions = predictions * (minmax_params[feature]["max"] - minmax_params[feature]["min"]) + minmax_params[feature]["min"]
    elif method == "minmax2":
        rescaled_predictions = (predictions + 1) * (minmax_params[feature]["max"] - minmax_params[feature]["min"]) / 2 + minmax_params[feature]["min"]
    elif method == "sin":
        rescaled_predictions = np.arcsin(predictions) / np.pi * 180.
    elif method == "standard":
        rescaled_predictions = predictions * minmax_params[feature]["std"] + minmax_params[feature]["mean"]
    else:
        rescaled_predictions = predictions * minmax_params[feature]["percentile_range"] + minmax_params[feature]["median"]
    return rescaled_predictions



if feature in intrabase_features:
    feature_count = len(pairs["A"])
else:
    feature_count = len(diPairs[("A", "A")])

@tf.function
def predict_step(inputs):
    predictions = model(inputs, training = False)
    #t_loss = loss_object(labels, predictions)
    #test_loss(t_loss)
    return predictions

def predict(seq, fout):
    if feature in interbase_features:
        x = oneHotDi(seq)
        rev = oneHotDi(revSeq(seq))
    else:
        x = oneHot(seq)
        rev = oneHot(revSeq(seq))
    x, pairs = preprocess_with_selfloop(x)
    kmers = [len(seq)]
    predictions = predict_step((x, pairs, kmers))

    rev, rev_pairs = preprocess_with_selfloop(rev)
    rev_predictions = predict_step((rev, rev_pairs, kmers))
    
    #if feature in interbase_features:
    #    predictions = bond_predictions
    predictions = rescale(predictions, minmax_params)
    rev_predictions = rescale(rev_predictions, minmax_params)

    if feature in x_axis_features:
        rev_predictions = -rev_predictions
    predictions = (predictions + rev_predictions[::-1]) / 2

    if padded:
        seq = seq[exclude_bp:-exclude_bp]
    if ifconstraints:
        for i, prediction in enumerate(tf.transpose(predictions)):
            if padded:
                prediction = prediction[exclude_bp:-exclude_bp]
            if if_self_layer:
                layerid = str(i)
            else:
                layerid = str(i + 1)
            fout.write(seq + " " + layerid + " ")
            np.savetxt(fout, prediction, newline = " ", fmt = "%.4f")
            fout.write("\n")
    else:
        if padded:
            predictions = predictions[exclude_bp:-exclude_bp]
        fout.write(seq + " ")
        np.savetxt(fout, predictions, newline = " ", fmt = "%.4f")
        fout.write("\n")

def predictBatch(seqBatch, fout):
    #seqBatch is list containing sequences
    #first step is to convert list of sequences into ragged encoded sequence tensor
    if feature in interbase_features:
        encodefunc = oneHotDi
    else:
        encodefunc = oneHot
    seqs = []
    revseqs = []
    for seq in seqBatch:
        seqs.append(encodefunc(seq))
        revseqs.append(encodefunc(revSeq(seq)))
    seqtensor = tf.ragged.constant(seqs, ragged_rank = 1)
    revseqtensor = tf.ragged.constant(revseqs, ragged_rank = 1)

    (x, pairs), kmers = prebatch_with_selfloop(seqtensor)
    predictions = rescale(predict_step((x, pairs, kmers)), minmax_params)
    (x, pairs), kmers = prebatch_with_selfloop(revseqtensor)
    rev_predictions = rescale(predict_step((x, pairs, kmers)), minmax_params)
    if feature in x_axis_features:
        rev_predictions = -rev_predictions

    predictions = tf.RaggedTensor.from_row_lengths(predictions, kmers)
    rev_predictions = tf.RaggedTensor.from_row_lengths(rev_predictions, kmers)
    if ifconstraints:
        rev_predictions = rev_predictions[:, ::-1, :]
    else:
        rev_predictions = rev_predictions[:, ::-1]
    predictions = (predictions + rev_predictions) / 2
    
    #predictions has shape of (num of seqs, None, num of output)
    if ifconstraints:
        for i in range(len(seqBatch)):
            thisseq = seqBatch[i]
            if padded:
                thisseq = thisseq[exclude_bp:-exclude_bp]
            prediction = predictions[i]
            for j,predictionline in enumerate(tf.transpose(prediction)):
                if padded:
                    predictionline = predictionline[exclude_bp:-exclude_bp]
                if if_self_layer:
                    layerid = str(j)
                else:
                    layerid = str(j + 1)
                fout.write(thisseq + " " + layerid + " ")
                np.savetxt(fout, predictionline, newline = " ", fmt = "%.4f")
                fout.write("\n")
    else:
        if padded:
            predictions = predictions[:, exclude_bp:-exclude_bp]
        #fout.write(thisseq + " ")
        np.savetxt(fout, predictions.numpy(), newline = "\n", fmt = "%.4f")
        #fout.write("\n")
    



print(args.mp_layers, args.steps, \
    basefeatures, basestepfeatures, \
    filters, weight_decay, ifconstraints,\
     multiply, if_self_layer, feature_count, ~if_nobn, ~if_nogru,)
model = DNAModel(mp_layer = args.mp_layers, mp_steps = args.steps, \
    basefeatures = basefeatures, basestepfeatures = basestepfeatures, \
    filter_size = filters, weight_decay = weight_decay, constraints = ifconstraints,\
    multiply = multiply, selflayer = if_self_layer, input_features=feature_count, bn_layer = not if_nobn, gru_layer = not if_nogru,)
model.load_weights(os.path.join("models", featuremodel))
model.model().summary()
    


if seq:
    if padded:
        seq = "NN" + seq + "NN"
    fout = sys.stdout
    if outputfile:
        fout = open(outputfile, "w")
    predict(seq, fout)
    if outputfile:
        fout.close()
elif filename:
    with open(outputfile, "w") as fout:
        with open(filename) as fin:
            batch = []
            for line in fin:
                seq = line.strip().split()[0]
                if padded:
                    seq = "NN" + seq + "NN"
                if args.batch_size == 1:
                    predict(seq, fout)
                else:
                    batch.append(seq)
                    if len(batch) == args.batch_size:
                        predictBatch(batch, fout)
                        batch = []
            if len(batch) != 0:
                predictBatch(batch, fout)
                batch = []