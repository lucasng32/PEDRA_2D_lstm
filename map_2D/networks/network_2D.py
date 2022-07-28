from statistics import stdev
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

import sys


class FCQN:
    def __init__(self, X, prev_lstmstate):    # TODO: NOW, HARDCODED AS 52x52 action space, can it be more flexible?
        self.X = X
        self.prev_lstmstate = prev_lstmstate

        #print(self.X.shape)
        self.conv1 = self.conv(self.X, k=5, out=96, s=1, names="conv1", p='VALID')

        #print("SELF con1 shape", self.conv1.shape)
        self.maxpool1 = tf.nn.max_pool(self.conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
        #print("SELF maxpool1 shape", self.maxpool1.shape)

        self.conv2 = self.conv(self.maxpool1, k=5, out=64, s=1, names="conv2", p="VALID")
        #print("SELF con2 shape", self.conv2.shape)
        self.maxpool2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        #print("SELF maxpool1 shape", self.maxpool2.shape)

        self.conv3 = self.conv(self.maxpool2, k=3, out=64, s=1, names="conv3", p="SAME")
        #print("SELF con3 shape", self.conv3.shape)
        
        # Advantage Network
        self.advantage = self.conv(self.conv3, k=1, out=1, s=1, names="advantage", p='VALID')
        #print("SELF advantage shape", self.advantage.shape)

        # Value Network (Global max pooling)
        self.value = tf.nn.max_pool(self.conv3, ksize=[1, 52, 52, 1], strides=[1, 1, 1, 1], padding='VALID')        # HARDCODED GLOBAL MAX POOL SIZE
        #print("SELF value shape", self.value.shape)

        # Q value of each point action
        self.out = tf.reshape((self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, keep_dims=True)))[0, :, :, 0], [-1])  # flattened

        # LSTM cell
        self.output, self.lstmstate_out = self.lstmcustom(self.out, self.prev_lstmstate)
            
        
    @staticmethod
    def conv(input, k, out, s, p, names, trainable=True):
        W = tf.get_variable(name = names+"W", shape=[k, k, int(input.shape[3]), out], initializer=orthogonal_initializer(), trainable=trainable)
        b = tf.get_variable(name = names+"b", shape=[out], initializer=tf.truncated_normal_initializer(stddev=0.05), trainable=trainable)
        #print(W.shape)
        conv_kernel_1 = tf.nn.conv2d(input, W, [1, s, s, 1], padding=p)
        bias_layer_1 = tf.nn.bias_add(conv_kernel_1, b)

        return tf.nn.leaky_relu(bias_layer_1)

    @staticmethod
    def lstmcustom(input, lstm_state):

        lstmin = tf.reshape(input, [1, 52*52])
        lstm_state_unpacked = tf.unstack(lstm_state, axis = 0)
        lstm_state_tuples = tf.nn.rnn_cell.LSTMStateTuple(lstm_state_unpacked[0][0],lstm_state_unpacked[0][1])

        depth = int(lstmin.shape[-1])

        c, h = tf.split(value=lstm_state_tuples, num_or_size_splits=2, axis=0)
        #print(c.shape)
        c = tf.squeeze(c)
        #print(h.shape)

        W_xh = tf.get_variable('W_xh',[depth, 4 * depth],initializer=orthogonal_initializer(),trainable=True)
        W_hh = tf.get_variable('W_hh',[depth, 4 * depth],initializer=orthogonal_initializer(),trainable=True)

        bias = tf.get_variable('bias', [4 * depth],initializer=tf.truncated_normal_initializer(stddev=0.05), trainable=True)

        xh = tf.matmul(lstmin, W_xh)
        #print(xh.shape)
        hh = tf.matmul(h, W_hh)

        hidden = xh + hh + bias
        #print(hidden.shape)
        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(value=hidden, num_or_size_splits=4, axis=1)
        new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)
        new_h = tf.tanh(new_c) * tf.sigmoid(o)

        new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
        lstm_out_reshaped = tf.squeeze(new_h)
        lstm_state_out_reshaped = tf.reshape(new_state, [1, 2, int(input.shape[0])])
        return lstm_out_reshaped, lstm_state_out_reshaped

#orthogonal initiliazer probably better
def orthogonal(shape):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 0.05, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    return q.reshape(shape)

def orthogonal_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        return tf.constant(orthogonal(shape), dtype)
    return _initializer
