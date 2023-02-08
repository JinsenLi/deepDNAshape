import tensorflow as tf
import numpy as np
from tensorflow.python.keras import activations
from tensorflow.keras import regularizers

class DNAtoGraph(tf.keras.layers.Layer):
    #This layer transformes the input shape of [B, k, 4] to [B*k, 4] and outputs additional linkage array [B*k, 2]
    #k is variable. Input is ragged batched tensor.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.batch_size = input_shape[0]
        self.channel = input_shape[-1]
    def call(self, inputs):
        mergedInput = inputs.merge_dims(outer_axis = 0, inner_axis = 1).to_tensor()
        kmers = inputs.row_lengths() # [k1, k2, k3, ..., kb]
        num_linkages = tf.reduce_sum(kmers) - self.batch_size
        linkages = tf.range(num_linkages, dtype = tf.int64) #raw 
        linkages = tf.reshape(tf.repeat(linkages, 4), (-1, 4)) + tf.pad(tf.ones((num_linkages,2), dtype = tf.int64), ((0,0), (1,1))) #add all connections
        linkages = linkages + tf.reshape(tf.repeat(tf.repeat(tf.range(self.batch_size, dtype = tf.int64), (kmers - 1)), 4), (-1, 4)) #add skip to seperate DNAs
        linkages = tf.reshape(linkages, (-1, 2)) #reshape to linkage
        return mergedInput, linkages

class avgFeatures(tf.keras.layers.Layer):
    #This layer take means of input channels according to required output.
    def __init__(self, targetFeature = 1, filter_size = 64, **kwargs):
        super().__init__(**kwargs)
        self.targetFeature = targetFeature if targetFeature != 0 else 1
        self.channel = filter_size
    def build(self, input_shape):
        #shape is (B*N, C)
        #self.batch_size = input_shape[0]
        #self.channel = input_shape[-1]
        self.aggrefeaturenums = self.channel // self.targetFeature
        self.padding = tf.math.floormod(self.channel, self.targetFeature)
    def call(self, inputs):
        inputs = tf.pad(inputs, [(0,0), (0, self.padding)])
        inputs = tf.reshape(inputs, (-1, self.targetFeature, self.aggrefeaturenums))
        return tf.reshape(tf.reduce_mean(inputs, axis = -1), [-1])

class avgBimodalFeatures(tf.keras.layers.Layer):
    #This layer take means of input channels according to required output.
    def __init__(self, units = 1, **kwargs):
        super().__init__(**kwargs)
        self.units = 1
        self.targetFeature = self.units * 3

    def build(self, input_shape):
        #shape is (B*N, C)
        #self.batch_size = input_shape[0]
        self.channel = input_shape[-1]
        self.aggrefeaturenums = self.channel // self.targetFeature
        self.padding = tf.math.floormod(self.channel, self.targetFeature)

    def call(self, inputs):
        inputs = tf.pad(inputs, [(0,0), (0, self.padding)])
        inputs = tf.reshape(inputs, (-1, self.targetFeature, self.aggrefeaturenums))
        inputs = tf.reshape(tf.reduce_mean(inputs, axis = -1), [-1, 3, self.units])
        return inputs

class messagePassingConv(tf.keras.layers.Layer):
    def __init__(self, filters = 64, kernel_size = 1, activation = "relu", weight_decay = 0.0, \
            trainable = True, padded = False, multiply = False, steps = 1, bn_layer = True, grulayer = True):
        super(messagePassingConv, self).__init__(dtype = tf.float32, trainable = trainable)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activations.get(activation)
        self.weight_decay = weight_decay
        self.bn = tf.keras.layers.BatchNormalization(input_shape = (None, filters))
        self.if_gru = grulayer
        if grulayer:
            self.gru_layer = tf.keras.layers.GRUCell(filters)
        else:
            self.gru_layer = activations.get("sigmoid")
        self.padded = padded
        self.steps = steps
        self.bn_layer = bn_layer
        self.multiply = multiply

    def build(self, input_shape):
        #input_shape is (node features, node pairs to next, node pairs to prev) [(B * N, k), (None, 2), (None, 2)]
        #print(input_shape)
        #self.oldChannel = input_shape[0][-1]
        #self.pad_length = tf.math.maximum(0, self.filters - self.oldChannel)
        self.wNext = self.add_weight("Weight1", shape = [self.filters, self.filters], initializer='random_normal', 
                trainable=True, dtype = tf.float32, regularizer = regularizers.l2(self.weight_decay))
        self.wPrev = self.add_weight("Weight2", shape = [self.filters, self.filters], initializer='random_normal', 
                trainable=True, dtype = tf.float32, regularizer = regularizers.l2(self.weight_decay))
        self.b = self.add_weight("Bias1", shape = [1, self.filters], initializer='zeros', 
                trainable=True, dtype = tf.float32)
        if self.multiply:
            if self.multiply == "add":
                self.wNext_all = self.add_weight("Weight1_term2", shape = [self.filters, self.filters], initializer='random_normal', 
                    trainable=True, dtype = tf.float32, regularizer = regularizers.l2(self.weight_decay))
                self.wPrev_all = self.add_weight("Weight2_term2", shape = [self.filters, self.filters], initializer='random_normal', 
                    trainable=True, dtype = tf.float32, regularizer = regularizers.l2(self.weight_decay))
            self.b_all = self.add_weight("BiasAll", shape = [1, self.filters], initializer='zeros', 
                trainable=True, dtype = tf.float32)
        #self.bPrev = self.add_weight("Bias2", shape = [1, self.filters], initializer='zeros', 
        #        trainable=True, dtype = tf.float32, regularizer = regularizers.l2(self.weight_decay))

    def call(self, inputs, training = False):
        x, pairs, kmers = inputs
        pairsPrev, pairsNext = pairs
        prev_x = tf.gather(x, pairsPrev[:, -1])
        prev_sumx = tf.math.segment_sum(prev_x, pairsPrev[:, 0])# + x
        next_x = tf.gather(x, pairsNext[:, -1])
        next_sumx = tf.math.segment_sum(next_x, pairsNext[:, 0])# + x
        aggre = tf.matmul(next_sumx, self.wNext) + tf.matmul(prev_sumx, self.wPrev) + self.b
        if self.padded:
            #shape of x is (B * N, k), pair_indices is (B * Edges, 2)
            
            #tf.tensor_scatter_nd_update(x_copy, tf.expand_dims(pairsCur[:, -1], axis = 1), aggre)
            #aggre = x_copy
            xshort = tf.RaggedTensor.from_row_lengths(x, kmers + 2)[:, 1:-1].merge_dims(0, 1)
            aggre = aggre + xshort
            aggre = self.bn(self.activation(aggre), training = training)
            xshort, _ = self.gru_layer(aggre, xshort)
            return tf.concat([tf.zeros((len(kmers), 1, self.filters)), tf.RaggedTensor.from_row_lengths(xshort, kmers), tf.zeros((len(kmers), 1, self.filters))], axis = 1).merge_dims(0,1)
        else:
            #next_x = tf.gather(x, pairs[:, -1])
            #next_sumx = tf.math.segment_sum(next_x, pairs[:, 0])# + x
            #next_sumx = next_sumx + x
            #aggre = tf.matmul(next_sumx, self.wNext) + self.b

            if self.multiply:
                if self.multiply == "add":
                    aggre = aggre + tf.matmul(next_sumx, self.wNext_all) + tf.matmul(prev_sumx, self.wPrev_all)
                aggre = aggre * x  + self.b_all
            else:
                aggre = aggre + x

            #aggre = next_aggre + prev_aggre
            if self.bn_layer:
                aggre = self.bn(self.activation(aggre), training = training)
            #return aggre
            if self.if_gru:
                x, _ = self.gru_layer(aggre, x)
            else:
                x = self.gru_layer(aggre)
            return x 

class messagePassingBondConv(tf.keras.layers.Layer):
    def __init__(self, filters = 64, kernel_size = 1, activation = "relu", weight_decay = 0.0, trainable = True):
        super(messagePassingBondConv, self).__init__(dtype = tf.float32, trainable = trainable)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activations.get(activation)
        self.weight_decay = weight_decay
        self.bn = tf.keras.layers.BatchNormalization()
        self.gru_layer = tf.keras.layers.GRUCell(filters)

    def build(self, input_shape):
        #input_shape is [(B * N, k), (B * N, k-1),  (None, 2)]
        #print(input_shape)
        #self.oldChannel = input_shape[1][-1]
        #self.pad_length = max(0, self.filters - self.oldChannel)
        self.wNext = self.add_weight("Weight1", shape = [self.filters, self.filters], initializer='random_normal', 
                trainable=True, dtype = tf.float32, regularizer = regularizers.l2(self.weight_decay))
        self.wPrev = self.add_weight("Weight2", shape = [self.filters, self.filters], initializer='random_normal', 
                trainable=True, dtype = tf.float32, regularizer = regularizers.l2(self.weight_decay))
        self.bNext = self.add_weight("Bias1", shape = [1, self.filters], initializer='zeros', 
                trainable=True, dtype = tf.float32, regularizer = regularizers.l2(self.weight_decay))
        self.bPrev = self.add_weight("Bias2", shape = [1, self.filters], initializer='zeros', 
                trainable=True, dtype = tf.float32, regularizer = regularizers.l2(self.weight_decay))

    def call(self, inputs, training = False):
        x, bond_x, pairsNext, pairsPrev = inputs #shape of x is (B * N, k), pair_indices is (B * Edges, 2)

        next_x = tf.gather(x, pairsNext[:, -1])
        prev_x = tf.gather(x, pairsPrev[:, -1])
        #next_sumx = tf.math.segment_sum(next_x, pairsNext[:, 0])# + x
        #prev_sumx = tf.math.segment_sum(prev_x, pairsPrev[:, 0])
        next_sumx = next_x + bond_x
        prev_sumx = prev_x + bond_x
        next_aggre = tf.matmul(next_sumx, self.wNext) + self.bNext
        prev_aggre = tf.matmul(prev_sumx, self.wPrev) + self.bPrev
        aggre = next_aggre + prev_aggre
        aggre = self.bn(self.activation(aggre), training = training)
        bond_x, _ = self.gru_layer(aggre, bond_x)
        return bond_x 

class DNANetwork(tf.keras.layers.Layer):
    #This network
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        pass
    def call(self, inputs):
        pass

class DNAModel(tf.keras.Model):
    def __init__(self, batch_size = 256, filter_size = 32, mp_layer = 1, mp_steps = 10, \
            basefeatures = 1, basestepfeatures = 0, weight_decay = 0, auto_weight_decay = False, \
            constraints = False, padded = False, multiply = False, selflayer = False, dropout_rate = 0.0, bn_layer = True, gru_layer = True, input_features = 15, **kwargs):
        #constraints mean that if you want to have a loss calculated for each message passing layer
        super().__init__(**kwargs)
        #self.preprocess_layer = DNAtoGraph()
        self.mp_layers_count = mp_layer
        self.batch_size = batch_size
        self.input_features = input_features
        self.steps = mp_steps
        self.mp_layers = []
        #self.mp_bond_layers = []
        self.num_basefeatures = basefeatures
        self.filter_size = filter_size
        self.padded = padded
        self.dropout_rate = dropout_rate
        #self.num_basestepfeatures = basestepfeatures
        self.selfconv = tf.keras.layers.Conv1D(filter_size, 1)
        #self.selfbn = tf.keras.layers.BatchNormalization()
        self.selflayer = selflayer
        self.bn_layer = bn_layer
        for _ in range(mp_layer):
            if auto_weight_decay:
                weight_decay = weight_decay * 10
            self.mp_layers.append(messagePassingConv(filters = filter_size, weight_decay = weight_decay, padded = False, multiply = multiply, bn_layer = bn_layer, grulayer = gru_layer))
            #if self.num_basestepfeatures > 0:
            #    self.mp_bond_layers.append(messagePassingBondConv(filters = filter_size))
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        self.avg_layer = avgFeatures(self.num_basefeatures, filter_size = filter_size)
        self.constraints = constraints
        #self.avg_bond_layer = avgFeatures(self.num_basestepfeatures)
    
    def callAvg(self, x, training = False):
        if training:
            x = self.dropout_layer(x)
        return self.avg_layer(x)

    def call(self, inputs, training = False):
        #inputs is (B, k, 3) ragged tensor
        x, pairs, kmers = inputs
        x = tf.expand_dims(x, axis = 0)
        x = self.selfconv(x)
        #if self.bn_layer:
        #    x = self.selfbn(x)
        x = tf.squeeze(x, axis = 0)
        #pad_length = tf.math.maximum(0, self.filter_size - tf.shape(x)[-1])
        #x = tf.pad(x, [(0,0), (0, pad_length)])
        #bond_pad_length = tf.math.maximum(0, self.filter_size - tf.shape(bond_x)[-1])
        #bond_x = tf.pad(bond_x, [(0,0), (0, bond_pad_length)])
        #x, pairs = self.preprocess_layer(x)
        #print(x, pairs)
        if self.selflayer:
            results = [self.callAvg(x, training)]
        else:
            results = []
        for i in range(len(self.mp_layers)):
            l = self.mp_layers[i]
            #l_bond = self.mp_bond_layers[i]
            for _ in range(self.steps):
                #bond_x = l_bond((x, bond_x, bondPairsNext, bondPairsPrev), training = training)
                x = l((x, pairs, kmers), training = training)
            if self.constraints:
                results.append(self.callAvg(x, training))
            #x, _ = self.gru_layer(x_next, x)
        if self.constraints:
            return tf.stack(results, axis = 1)
        return self.callAvg(x, training)
        #return tf.concat((self.avg_layer(x), self.avg_bond_layer(bond_x)), axis = 0)

    def model(self):
        x = tf.keras.Input(shape=(self.input_features), dtype = tf.float32)
        pairs1 = tf.keras.Input(shape=(2), dtype = tf.int64)
        pairs2 = tf.keras.Input(shape=(2), dtype = tf.int64)
        kmers = tf.keras.Input(shape = (0,), batch_size = self.batch_size, dtype = tf.int64)
        return tf.keras.Model(inputs=[x, (pairs1, pairs2), kmers], outputs=self.call((x, (pairs1, pairs2), kmers)))