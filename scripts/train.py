import argparse
parser = argparse.ArgumentParser(description='Train DNAfold.')
parser.add_argument("--trainfiles", dest = "trainfiles", nargs = "+", help = "The filepath to the train file.")
parser.add_argument("--valfile", dest = "valfile", help = "set if you want run validation in parallel.")
parser.add_argument("--feature", dest = "feature", help = "Features to be used in training.")
parser.add_argument("--steps", dest = "steps", default = 10, type = int)
parser.add_argument("--model_path", dest = "model_path", default="./models/")
parser.add_argument("--mp_layers", dest = "mp_layers", default = 1, type = int)
parser.add_argument("--batch_size", dest = "batch_size", default = 256, type = int)
parser.add_argument("--filters", dest = "filters", default = 64, type = int)
parser.add_argument("--epochs", dest = "epochs", default = 10000, type = int)
parser.add_argument("--lr", dest = "lr", default = 0.01, type = float)
parser.add_argument("--weight_decay", dest = "weight_decay", default = 0.0, type = float)
parser.add_argument("--auto_weight_decay", dest = "auto_weight_decay", action = "store_true")
parser.add_argument("--exclude_bp", dest = "exclude_bp", default = 1, type = int)
parser.add_argument("--loss", dest = "lossfunction", default = "MSE", choices = ["MSE", "MAE", "Huber"])
parser.add_argument("--constraints", dest = "constraints", action = "store_true")
parser.add_argument("--multiply", dest = "multiply", action = "store_true", help = "when selected, change the model to use 'multiply' rather than 'add'")
parser.add_argument("--add", dest = "add", action = "store_true", help = "When select multiply, choose if you want to add another flavor of 'add'.")
parser.add_argument("--padding", dest = "padding", action = "store_true", help = "Activate this option will add 2 Ns to the terminal of sequences. For example, ACGTA will become NNACGTANN in training. [exclude_bp] will also increase by 2.")
parser.add_argument("--val", dest = "if_validation", action = "store_true", help = "Select if you want to perform validation on training.")
parser.add_argument("--self_layer", dest = "if_self_layer", action = "store_true")
parser.add_argument("--dropout_rate", dest = "dropout_rate", type = float, default=0.0)
parser.add_argument("--optimizer", choices=["Adam", "SGD"], default="Adam")
parser.add_argument("--5mc", dest = "if_5mc", action = "store_true")
parser.add_argument("--phychem_encoding", dest = "if_phychem", action = "store_true")
parser.add_argument("--nobn", dest = "if_nobn", action = "store_true")
parser.add_argument("--nogru", dest = "if_nogru", action = "store_true", help = "If set, replace gru cell with a sigmoid function.")
args = parser.parse_args()

import tensorflow as tf
import numpy as np
from functools import partial
import sys, os
from model import DNAModel
import itertools
import model_utils
#import tensorflow_probability as tfp

pairs, diPairs = model_utils.getBasesMapping(args.if_phychem, args.if_5mc)
def oneHot(seq, padding = False):
    #if padding:
    #    seq = "N" + seq + "N"
    return np.array(list(map(lambda x: pairs[x], seq)))
def oneHotDi(seq, padding = False):
    #if padding:
    #    seq = "N" + seq + "N"
    return np.array(list(map(lambda i: diPairs[(seq[i], seq[i+1])], range(len(seq)-1))))
def oneHotDi2(seq, padding = False):
    #if padding:
    #    seq = "N" + seq + "N"
    return np.array(list(map(lambda i: pairs[seq[i]] + pairs[seq[i+1]], range(len(seq)-1))))

def loadData(filenames):
    fhandle = []
    for filename in filenames:
        fhandle.append(open(filename, "r"))
    while 1:
        curSeq = ""
        features = []
        eof = False
        for fin in fhandle:
            line = fin.readline()
            if not line:
                eof = True
                break
            items = line.strip().split()
            seq = items[0]
            if curSeq != "":
                assert(curSeq == seq)
            else:
                curSeq = seq
            features += list(map(float, items[1:]))
            if padded:
                seq = "NN" + seq + "NN"
                features = [float("nan"), float("nan")] + features + [float("nan"), float("nan")]
        #print(oneHot(seq), features)
        if eof:
            break
        yield tf.ragged.constant(oneHot(seq, padded), dtype = tf.float32), tf.ragged.constant(oneHotDi(seq, padded), dtype = tf.float32), tf.ragged.constant(np.array(features)[tf.newaxis], dtype = tf.float32)
    for fin in fhandle:
        fin.close()


'''
@tf.function
def train_step_bimodal(inputs, labels, kmers):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training = True) #predictions are (N, 3, units) where (:, 0, :) is mu (:, 1, :) is sigma, (:, 2, :) is phi
        mu, var, pi = predictions[:, 0, :], predictions[:, 1, :], predictions[:, 2, :]
        likelihood = tfp.distributions.Normal(loc=mu, scale=var)
        out = likelihood.prob(labels)
        out = tf.multiply(out, pi)
        out = tf.reduce_sum(out, 1, keepdims=True)
        out = -tf.log(out + 1e-10)
        loss = tf.reduce_mean(out)

        #labels = tf.RaggedTensor.from_row_lengths(labels, kmers) 
        #labels = labels[:, 1:-1]
        #labels = labels.merge_dims(0, 1)
        #predictions = tf.RaggedTensor.from_row_lengths(predictions, kmers)
        #predictions = predictions[:, 1:-1]
        #predictions = predictions.merge_dims(0, 1)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    return predictions
'''



def prebatch(inputs, label):
    mergedInput = inputs.merge_dims(outer_axis = 0, inner_axis = 1).to_tensor()
    batch_size = inputs.shape[0]
    kmers = inputs.row_lengths() # [k1, k2, k3, ..., kb]
    num_linkages = tf.reduce_sum(kmers) - batch_size
    linkages = tf.range(num_linkages, dtype = tf.int64) #raw 
    linkages = tf.reshape(tf.repeat(linkages, 4), (-1, 4)) + tf.pad(tf.ones((num_linkages,2), dtype = tf.int64), ((0,0), (1,1))) #add all connections
    linkages = linkages + tf.reshape(tf.repeat(tf.repeat(tf.range(batch_size, dtype = tf.int64), (kmers - 1)), 4), (-1, 4)) #add skip to seperate DNAs
    linkages = tf.reshape(linkages, (-1, 2)) #reshape to linkage
    return (mergedInput, linkages), label.merge_dims(0, 2), kmers


def prebatch_with_padding_directional(inputs, label):
    mergedInput = inputs.merge_dims(outer_axis = 0, inner_axis = 1).to_tensor()
    batch_size = inputs.shape[0]
    kmers = inputs.row_lengths() - 2 # [k1, k2, k3, ..., kb]
    num_linkages = tf.reduce_sum(kmers)
    linkages = tf.range(1, num_linkages + 1, dtype = tf.int64) #raw 
    linkages = tf.reshape(tf.repeat(linkages, 6), (-1, 6)) # (1,1,1,1,1,1), (2,2,2,2,2,2) ... (n,n,n,n,n,n)
    linkages = linkages + tf.pad(tf.ones((num_linkages,1), dtype = tf.int64), ((0,0), (5,0))) - tf.pad(tf.ones((num_linkages,1), dtype = tf.int64), ((0,0), (1,4))) #add all connections
    #linkages is not (1,0,1,1,1,2), (2,1,2,2,2,3), ... (n,n-1,n,n,n,n+1)
    linkages = linkages + tf.reshape(tf.repeat(tf.range(batch_size, dtype = tf.int64) * 2, kmers * 6), (-1, 6)) #add skip to seperate DNAs
    linkagesPrev = linkages[:, :2]
    linkagesCurr = linkages[:, 2:4]
    linkagesNext = linkages[:, 4:]
    return (mergedInput, (linkagesPrev, linkagesNext)), label.merge_dims(0, 2), kmers

def prebatch_with_selfloop(inputs, label):
    mergedInput = inputs.merge_dims(outer_axis = 0, inner_axis = 1).to_tensor()
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
    return (mergedInput, (linkagesPrev, linkagesNext)), label.merge_dims(0, 2), kmers


def prebatch_with_bond(inputs, label):
    mergedInput = inputs.merge_dims(outer_axis = 0, inner_axis = 1).to_tensor()
    batch_size = inputs.shape[0]
    kmers = inputs.row_lengths() # [k1, k2, k3, ..., kb]
    num_linkages = tf.reduce_sum(kmers) - batch_size
    bond_features = tf.zeros((num_linkages, 1), dtype = tf.float32)
    linkages = tf.range(num_linkages, dtype = tf.int64) #raw 
    linkages = tf.reshape(tf.repeat(linkages, 4), (-1, 4)) # make (0,0,0,0),(1,1,1,1)
    
    #bond_linkages have shape of (None, 2) where each (i, j) in the array means the linkage between bond i and node j.
    bond_linkages = linkages + tf.pad(tf.ones((num_linkages,1), dtype = tf.int64), ((0,0), (3,0))) #add connections
    bond_linkages = tf.reshape(bond_linkages, (-1, 2))
    padding = tf.pad(tf.reshape(tf.repeat(tf.range(batch_size, dtype = tf.int64), (kmers - 1) * 2), (-1, 1)), ((0,0), (1,0)))
    bond_linkages = bond_linkages + padding
    bond_linkages = tf.reshape(bond_linkages, (-1, 4))
    bondLinkagesPrev = bond_linkages[:, :2]
    bondLinkagesNext = bond_linkages[:, 2:]

    linkages = linkages + tf.pad(tf.ones((num_linkages,2), dtype = tf.int64), ((0,0), (1,1))) #add all connections
    linkages = linkages + tf.reshape(tf.repeat(tf.repeat(tf.range(batch_size, dtype = tf.int64), (kmers - 1)), 4), (-1, 4)) #add skip to seperate DNAs
    linkages = tf.reshape(linkages, (-1, 2)) #reshape to linkage
    return (mergedInput, linkages, bond_features, bondLinkagesNext, bondLinkagesPrev), label.merge_dims(0, 2), kmers

intrabase_features = {"Shear", "Stretch", "Stagger", "Buckle", "ProT", "Opening", "MGW", "EP", "Shear-FL", "Stretch-FL", "Stagger-FL", "Buckle-FL", "ProT-FL", "Opening-FL", "MGW-FL",  "EP-FL"}
interbase_features = {"Shift", "Slide", "Rise", "Tilt", "Roll", "HelT", "Shift-FL", "Slide-FL", "Rise-FL", "Tilt-FL", "Roll-FL", "HelT-FL"}

learning_rate = args.lr
epsilon = 1e-3
momentum = 0.95
batch_size = args.batch_size
EPOCHS = args.epochs
filters = args.filters
feature = args.feature
filenames = args.trainfiles
weight_decay = args.weight_decay
if_nobn = args.if_nobn
exclude_bp = args.exclude_bp
lossfunction = args.lossfunction
if_nogru = args.if_nogru
ifconstraints = args.constraints
mp_layers = args.mp_layers
optimizer_choice = args.optimizer
auto_weight_decay = args.auto_weight_decay
padded = args.padding
valfile = args.valfile
if padded:
    exclude_bp += 2
multiply = args.multiply
ifadd = args.add
model_path = args.model_path
if_self_layer = args.if_self_layer
dropout_rate = args.dropout_rate
if multiply and ifadd:
    multiply = "add"
initial_feature_count = len(pairs["A"])
initial_difeature_count = len(diPairs[("A", "A")])
if feature in intrabase_features:
    feature_count = initial_feature_count
else:
    feature_count = initial_difeature_count
#basestepfeatures = 0
#basefeatures = 0
#for feature in features:
##    basestepfeatures += 1 if feature in interbase_features else 0
#    basefeatures += 1 if feature in intrabase_features else 0


@tf.function(
    input_signature=[tf.TensorSpec(shape=(None, feature_count), dtype=tf.float32),
                    (tf.TensorSpec(shape=(None, 2), dtype=tf.int64), tf.TensorSpec(shape=(None, 2), dtype=tf.int64)),
                    tf.TensorSpec(shape=(None), dtype=tf.float32),
                    tf.TensorSpec(shape=(None), dtype=tf.int64)])
def train_step(x, pairs, labels, kmers):
    #x, pairs = original_inputs
    inputs = (x, pairs, kmers)
    with tf.GradientTape() as tape:
        predictions = model(inputs, training = True)
        #if padded:
        #    predictions = tf.RaggedTensor.from_row_lengths(predictions, kmers + 2)
        #    predictions = predictions[:, 1:-1]
        #    predictions = predictions.merge_dims(0, 1)
        if ifconstraints:
            #predictions have shape (n, k), n is number of nodes, k is number of layers
            labels = tf.reshape(tf.repeat(labels, mp_layers + if_self_layer), (-1, mp_layers + if_self_layer)) # reshape to (n, k)
        if exclude_bp > 0:
            labels = tf.RaggedTensor.from_row_lengths(labels, kmers)
            labels = labels[:, exclude_bp:-exclude_bp]
            labels = labels.merge_dims(0, 1)
            predictions = tf.RaggedTensor.from_row_lengths(predictions, kmers)
            predictions = predictions[:, exclude_bp:-exclude_bp]
            predictions = predictions.merge_dims(0, 1)
        labels = tf.where(tf.math.is_nan(labels), predictions, labels)
        labels = tf.where(tf.math.greater(labels , 1.0), 1.0, labels)
        labels = tf.where(tf.math.less(labels, -1.0), -1.0, labels)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    return predictions

#no exclusion on validation
@tf.function(
    input_signature=[tf.TensorSpec(shape=(None, feature_count), dtype=tf.float32),
                    (tf.TensorSpec(shape=(None, 2), dtype=tf.int64), tf.TensorSpec(shape=(None, 2), dtype=tf.int64)),
                    tf.TensorSpec(shape=(None), dtype=tf.float32),
                    tf.TensorSpec(shape=(None), dtype=tf.int64)])
def test_step(x, pairs, labels, kmers):
    #x, pairs = inputs
    inputs = (x, pairs, kmers)
    predictions = model(inputs, training = False)
    if ifconstraints:
        #predictions have shape (n, k), n is number of nodes, k is number of layers
        labels = tf.reshape(tf.repeat(labels, mp_layers + if_self_layer), (-1, mp_layers + if_self_layer)) # reshape to (n, k)
    #if exclude_bp > 0:
    #    labels = tf.RaggedTensor.from_row_lengths(labels, kmers)
    #    labels = labels[:, exclude_bp:-exclude_bp]
    #    labels = labels.merge_dims(0, -1)
    #    predictions = tf.RaggedTensor.from_row_lengths(predictions, kmers)
    #    predictions = predictions[:, exclude_bp:-exclude_bp]
    #    predictions = predictions.merge_dims(0, -1)
    labels = tf.where(tf.math.is_nan(labels), predictions, labels)
    labels = tf.where(tf.math.greater(labels , 1.0), 1.0, labels)
    labels = tf.where(tf.math.less(labels, -1.0), -1.0, labels)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    return labels, predictions # shape of (k) or (k, 10)

model = DNAModel(mp_layer = mp_layers, mp_steps = args.steps, basefeatures = 1, \
        filter_size = filters, weight_decay = weight_decay, auto_weight_decay = auto_weight_decay, constraints = ifconstraints, padded = padded,\
        multiply = multiply, selflayer = if_self_layer, dropout_rate = dropout_rate, bn_layer = not if_nobn, gru_layer = not if_nogru, batch_size = batch_size, input_features=feature_count)
model.model().summary()
if optimizer_choice == "Adam":
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate, epsilon = epsilon)
elif optimizer_choice == "SGD":
    optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum = momentum)
if lossfunction == "MSE":
    loss_object = tf.keras.losses.MeanSquaredError()
elif lossfunction == "MAE":
    loss_object = tf.keras.losses.MeanAbsoluteError()
else:
    loss_object = tf.keras.losses.Huber()
train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')
modelpath = os.path.join(model_path, feature)
ds = tf.data.Dataset.from_generator(
    partial(loadData, filenames),
    output_signature = (
        tf.RaggedTensorSpec(shape = (None, initial_feature_count), dtype = tf.float32),
        tf.RaggedTensorSpec(shape = (None, initial_difeature_count), dtype = tf.float32),
        tf.RaggedTensorSpec(shape = (None, 1), dtype = tf.float32)
    )
)
ds = ds.cache().shuffle(batch_size * 4).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
if valfile:
    valds = tf.data.Dataset.from_generator(
        partial(loadData, [valfile]),
        output_signature = (
            tf.RaggedTensorSpec(shape = (None, initial_feature_count), dtype = tf.float32),
            tf.RaggedTensorSpec(shape = (None, initial_difeature_count), dtype = tf.float32),
            tf.RaggedTensorSpec(shape = (None, 1), dtype = tf.float32)
        )
    )
    valds = valds.cache().batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


fsnapshot = open(modelpath + ".txt", "w")
if ifconstraints:
    fsnapshot.write("Epochs\tLoss\t")
    if valfile:
        if if_self_layer:
            fsnapshot.write("ValLoss_self\t")
        for i in range(mp_layers):
            fsnapshot.write("ValLoss_" + str(i + 1) + "\t")
    fsnapshot.write("\n")
else:
    fsnapshot.write("Epochs\tLoss\tValLoss\n")
#if padded:
#    prebatch_func = prebatch_with_padding_directional
#else:
prebatch_func = prebatch_with_selfloop
for epoch in range(EPOCHS):
    for monoinputs, diinputs, dslabels in ds:
        if feature in interbase_features:
            inputs, labels, kmers = prebatch_func(diinputs, dslabels)
        else:
            inputs, labels, kmers = prebatch_func(monoinputs, dslabels)
        #inputs, labels, kmers = prebatch_with_bond(inputs, labels)
        #print(inputs)
        predictions = train_step(inputs[0], inputs[1], labels, kmers)
        #print(predictions)
    template = 'Epoch {}, Loss: {}'
    print(template.format(epoch+1,
                        train_loss.result()))

    #if valfile and epoch % 5 == 4:
    if valfile:
        val_labels = []
        val_predicitons = []
        for monoinputs, diinputs, dslabels in valds:
            if feature in interbase_features:
                inputs, labels, kmers = prebatch_func(diinputs, dslabels)
            else:
                inputs, labels, kmers = prebatch_func(monoinputs, dslabels)
            #inputs, labels, kmers = prebatch_with_bond(inputs, labels)
            #print(inputs)
            batch_labels, batch_predictions = test_step(inputs[0], inputs[1], labels, kmers)
            val_labels.append(batch_labels.numpy())
            val_predicitons.append(batch_predictions.numpy())
        val_labels = np.concatenate(val_labels, axis = 0)
        val_predicitons = np.concatenate(val_predicitons, axis = 0)
        template = '    Epoch {}, Val Loss: {}'
        print(template.format(epoch+1,
                        test_loss.result()))

    if ifconstraints:
        fsnapshot.write("%d\t%f" % (epoch + 1, train_loss.result()))
        if valfile:
            layer_labels = np.reshape(val_labels, (-1, mp_layers + if_self_layer))
            layer_predictions = np.reshape(val_predicitons,(-1, mp_layers + if_self_layer))
            for i in range(mp_layers + if_self_layer):
                fsnapshot.write("\t%f" % tf.keras.losses.mean_absolute_error(layer_labels[:, i], layer_predictions[:, i]))
        fsnapshot.write("\n")
    else:
        fsnapshot.write("%d\t%f\t" % (epoch + 1, train_loss.result()))
        if valfile:
            fsnapshot.write("%f\n" % (test_loss.result()))
    fsnapshot.flush()
    if (epoch + 1) % 100 == 0 or epoch == EPOCHS - 1:
        print("Saving weights.")
        model.save_weights(modelpath, overwrite=True)
    test_loss.reset_states()
    train_loss.reset_states()
fsnapshot.close()