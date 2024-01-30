from . import Model
from . import Model_utils
import os
import numpy as np
import tensorflow as tf
import json
import pkgutil


class predictor:
    def __init__(self, mode = "cpu"):
        if mode == "cpu":
            tf.config.set_visible_devices([], 'GPU')
        self.models = {}
        self.pairs, self.diPairs = Model_utils.getBasesMapping(False, False)
        self.x_axis_features = set(['Shift', 'Tilt', 'Shear', 'Buckle'])
        json_file = pkgutil.get_data(__name__, "params.json")
        self.minmax_params = json.loads(json_file)
        self.intrabase_features = {"Shear", "Stretch", "Stagger", "Buckle", "ProT", "Opening", "MGW", "EP", "Shear-FL", "Stretch-FL", "Stagger-FL", "Buckle-FL", "ProT-FL", "Opening-FL", "MGW-FL"}
        self.interbase_features = {"Shift", "Slide", "Rise", "Tilt", "Roll", "HelT", "Shift-FL", "Slide-FL", "Rise-FL", "Tilt-FL", "Roll-FL", "HelT-FL"}

    def oneHot(self, seq, padding = False):
        return np.array(list(map(lambda x: self.pairs[x], seq)))
    def revSeq(self, seq):
        revpairs = {"A":"T", "T":"A", "C":"G", "G":"C", "N": "N", "M": "g", "g": "M"}
        return "".join(list(map(lambda x: revpairs[x], seq))[::-1])
    def oneHotDi(self, seq, padding = False):
        return np.array(list(map(lambda i: self.diPairs[(seq[i], seq[i+1])], range(len(seq)-1))))
    def oneHotDi2(self, seq, padding = False):
        return np.array(list(map(lambda i: self.pairs[seq[i]] + self.pairs[seq[i+1]], range(len(seq)-1))))
    def rescale(self, feature, predictions):
        method = self.minmax_params[feature]["method"]
        if method == "minmax":
            rescaled_predictions = predictions * (self.minmax_params[feature]["max"] - self.minmax_params[feature]["min"]) + self.minmax_params[feature]["min"]
        elif method == "minmax2":
            rescaled_predictions = (predictions + 1) * (self.minmax_params[feature]["max"] - self.minmax_params[feature]["min"]) / 2 + self.minmax_params[feature]["min"]
        elif method == "sin":
            rescaled_predictions = np.arcsin(predictions) / np.pi * 180.
        elif method == "standard":
            rescaled_predictions = predictions * self.minmax_params[feature]["std"] + self.minmax_params[feature]["mean"]
        else:
            rescaled_predictions = predictions * self.minmax_params[feature]["percentile_range"] + self.minmax_params[feature]["median"]
        return rescaled_predictions

    def preprocess_with_selfloop(self, x):
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

    def prebatch_with_selfloop(self, inputs):
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
        return mergedInput, (linkagesPrev, linkagesNext), kmers

    def loadmodel(self, feature):
        input_feature_count = 4 if feature in self.intrabase_features else 16
        model = Model.DNAModel(mp_layer = 7, mp_steps = 1,
                    basefeatures = 1, basestepfeatures = 0, 
        filter_size = 64, weight_decay = 0, constraints = True,
        multiply = "add", selflayer = True, input_features=input_feature_count, bn_layer = True, gru_layer = True)
        model.load_weights(os.path.join(os.path.dirname(__file__), "models", feature))
        model.model()
        self.models[feature] = model
    def loadAll(self):
        for feature in list(self.intrabase_features) + list(self.interbase_features):
            self.loadmodel(feature)

    def predict_step(self, model, inputs):
        predictions = model(inputs, training = False)
        return predictions

    def predict(self, feature, seq, layer = None):
        #seq will be automatically padded with NN on both terminals, and removed in the predictions
        if layer:
            assert(0 <= layer <= 7)
        seq = "NN" + seq + "NN"
        if feature not in self.models:
            self.loadmodel(feature)
        model = self.models[feature]
        revseq = self.revSeq(seq)
        if feature in self.interbase_features:
            encodeFunc = self.oneHotDi
        else:
            encodeFunc = self.oneHot
        x = encodeFunc(seq)
        revx = encodeFunc(revseq)
        x, ipairs = self.preprocess_with_selfloop(x)
        revx, revipairs = self.preprocess_with_selfloop(revx)
        kmers = [len(seq)]
        predictions = self.rescale(feature, self.predict_step(model, (x, ipairs, kmers)))
        rev_predictions = self.rescale(feature, self.predict_step(model, (revx, revipairs, kmers)))
        if feature in self.x_axis_features:
            rev_predictions = -rev_predictions
        predictions = (predictions + rev_predictions[::-1]) / 2
        predictions = tf.transpose(predictions)
        #if layer and type(layer) is int and 0 <= layer <= 7:
        predictions = predictions[layer]
        return predictions.numpy()[2:-2]

    def predictBatch(self, feature, seqBatch, layer = None):
        #seq will be automatically padded with NN on both terminals, and removed in the predictions
        if feature not in self.models:
            self.loadmodel(feature)
        model = self.models[feature]
        if feature in self.interbase_features:
            encodeFunc = lambda x: self.oneHotDi("NN" + x + "NN")
        else:
            encodeFunc = lambda x: self.oneHot("NN" + x + "NN")
        x = list(map(encodeFunc, seqBatch))
        revx = list(map(lambda x: encodeFunc(self.revSeq(x)), seqBatch))
        x = tf.ragged.constant(x, ragged_rank = 1)
        revx = tf.ragged.constant(revx, ragged_rank = 1)
        x, ipairs, kmers = self.prebatch_with_selfloop(x)
        predictions = self.rescale(feature, self.predict_step(model, (x, ipairs, kmers)))
        predictions = tf.RaggedTensor.from_row_lengths(predictions, kmers)

        revx, revipairs, kmers = self.prebatch_with_selfloop(revx)
        rev_predictions = self.rescale(feature, self.predict_step(model, (revx, revipairs, kmers)))
        if feature in self.x_axis_features:
            rev_predictions = -rev_predictions
        rev_predictions = tf.RaggedTensor.from_row_lengths(rev_predictions, kmers)
        rev_predictions = rev_predictions[:, ::-1, :]

        predictions = (predictions + rev_predictions) / 2
        #if layer and type(layer) is int and 0 <= layer <= 7:
        predictions = predictions[:, 2:-2, layer]
        return predictions.numpy()


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import argparse, sys
    programname = sys.argv[0]
    parser = argparse.ArgumentParser(description='Predict DNA shapes for any sequences. Input can be one single sequence or a FILE containing multiple sequences.\n\nFILE format: \nSEQ1\nSEQ2\nSEQ3\n...\n\nExamples:\npython '+programname+' --seq AAGGTAGT --feature MGW\
        \npython '+programname+'.py --file seq.txt --feature Roll --output seq_Roll.txt', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--feature", dest = "feature", default = "MGW", help = "Specify which DNA shape feature to be predicted. [Default is MGW (minor groove width), other options: Shear, Stretch, Stagger, Buckle, ProT, Opening, Shift, Slide, Rise, Tilt, Roll, HelT]. Use FEATURE-FL to predict fluctuation values.")
    parser.add_argument("--seq", dest = "seq", help = "Specify the sequence to be predicted.")
    parser.add_argument("--file",dest = "file", help = "If no --seq is provided, use --file to read in all sequences.")
    parser.add_argument("--layer", default=4, dest = "layer", type = int, help = "Select output layer number (0 to 7). Choose bigger number if you want to evaluate longer range effects from flanking regions. Don't change this parameter if you are not sure. [Default is 4].")
    parser.add_argument("--output", dest = "output", default = "stdout", help = "Specify where the predictions will be written to. [Defualt is stdout]")
    parser.add_argument("--batch_size", dest = "batch_size", default=2048, type = int, help = "If --file is provided, use this parameter to adjust parallel computing maximum. [Default is 2048] use higher values to speed up your prediction subject to your CPU/mem resources.")
    parser.add_argument("--gpu", dest = "gpu", help = "Use --gpu if you have available GPUs, and are predicting a file, make sure CUDA, CuDNN are installed correctly.")
    args = parser.parse_args()

    myPredictor = predictor()
    if args.seq:
        prediction = list(map(str, myPredictor.predict(args.feature, args.seq, args.layer)))
        print(args.seq + " " + " ".join(prediction))
    else:
        if args.output == "stdout":
            fout = sys.stdout
        else:
            fout = open(sys.output, "w")
        with open(args.file) as fin:
            seqBatch = []
            for seq in fin:
                seqBatch.append(seq.strip())
                if len(seqBatch) == args.batch_size:
                    prediction = myPredictor.predictBatch(args.feature, seqBatch, args.layer)
                    fout.write(prediction.shape)
                    for i, seq in enumerate(seqBatch):
                        fout.write(seq + " " + " ".join(list(map(str, prediction[i]))) + "\n")
                    seqBatch = []
            if seqBatch != []:
                prediction = myPredictor.predictBatch(args.feature, seqBatch, args.layer)
                for i, seq in enumerate(seqBatch):
                    fout.write(seq + " " + " ".join(list(map(str, prediction[i]))) + "\n")
        fout.close()
