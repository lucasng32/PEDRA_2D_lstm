import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
from map_2D.networks.loss_functions import huber_loss
from map_2D.networks.network_2D import FCQN
import sys


class initialize_network_FCQN:
    def __init__(self, plot_directory: str, save_weights_dir: str, custom_load_path=None):
        self.g = tf.Graph()

        with self.g.as_default():
            stat_writer_path = plot_directory
            self.save_weights_path = save_weights_dir
            self.stat_writer = tf.summary.FileWriter(stat_writer_path)

            self.lstm_init_state = tf.Variable(name="nn", initial_value = tf.truncated_normal(shape = [1, 2, 52**2], stddev=0.05), trainable = False)

            self.input_size = 224

            # Placeholders
            self.batch_size = tf.placeholder(tf.int32, shape=())
            self.learning_rate = tf.placeholder(tf.float32, shape=())
            self.lstm_state = tf.Variable(name="state", initial_value = tf.truncated_normal(shape = [1, 2, 52**2], stddev=0.05), trainable = True)

            self.sampled_grids = tf.placeholder(tf.float32, [None, 224, 224, 1], name='BHM_Linearly_Sampled_Grids')
            self.current_position_map = tf.placeholder(tf.float32, [None, 224, 224, 1], name='Current_pos_map')
            self.previous_position_map = tf.placeholder(tf.float32, [None, 224, 224, 1], name='Previous_pos_map')
            self.X = tf.concat((self.sampled_grids, self.current_position_map, self.previous_position_map), axis=3, name='Combined_image')

            # target will be fed later as Qvals of next state (which is Reward gained + discount * max(Q of next s a) - Q of current s a)
            self.target = tf.placeholder(tf.float32, shape=[None],      name='Target_Qs')
            self.action = tf.placeholder(tf.int32, shape=[None, 2],   name='Actions')

            self.model = FCQN(self.X, self.lstm_state)
            self.action_space = 52 ** 2
            self.predict = self.model.output    # for simplicity, i flattened my model output to 2704 (52 ** 2)
            self.lstm_state_output = self.model.lstmstate_out
            action_flattened_index = self.action[0][0] * 52 + self.action[0][1]
            # self.actions = np.ravel_multi_index(self.actions, self.model.output.shape)
            # print('action:', self.action[0])
            # print('action flattened index:', action_flattened_index.shape)
            ind = tf.one_hot(action_flattened_index, self.action_space)
            
            self.restore_state = tf.assign(self.lstm_state,self.lstm_init_state)

            self.propagate_state = tf.assign(self.lstm_state,self.lstm_state_output)

            self.save_state = tf.assign(self.lstm_init_state,self.lstm_state)

            # self.flattened_output = tf.placeholder(tf.float32, shape=(None, 2704), name='Flattened_weights')
            #self.weight_test = self.model.weights_test

            # flattened_output = tf.reshape(self.model.output, [-1])
            # print('flattened_output', self.flattened_output)
            pred_Q = tf.reduce_sum(
                tf.multiply(self.model.output, ind),
                axis=0)
            # print('pred_Q', pred_Q)
            self.loss = huber_loss(pred_Q, self.target)     # original paper used MSE

            self.all_vars = tf.trainable_variables()

            self.nn_vars = [var for var in self.all_vars if 'state' not in var.name]

            self.train_nn = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.99).minimize(self.loss, name="train_nn", var_list=self.nn_vars)

            self.train_first_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.99).minimize(self.loss, name="train_step", var_list=self.all_vars)
            #opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.99, name='train')
            #gradients, variables = zip(*opt.compute_gradients(self.loss))
            #gradients_clipped, _ = tf.clip_by_global_norm(gradients, 3.0)
            #self.train = opt.apply_gradients(zip(gradients_clipped, variables))

            self.sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            self.saver = tf.train.Saver(self.all_vars)
            self.sess.graph.finalize()


        # Load custom weights from custom_load_path if required
        if custom_load_path is not None:
            try:
                print('Loading weights from: ', custom_load_path)
                self.load_network(custom_load_path)
            except:
                pass

    def get_vars(self):
        return self.sess.run(self.all_vars)

    def Q_val(self, curr_state_tuple, propagate_state, first_step):
        grids, curr_positions, prev_positions = curr_state_tuple
        if first_step: self.sess.run(self.restore_state)
        Q = self.sess.run(self.predict,
                             feed_dict={self.batch_size: grids.shape[0], self.sampled_grids: grids,
                                        self.current_position_map: curr_positions, self.previous_position_map: prev_positions, 
                                        })
        if propagate_state: self.sess.run(self.propagate_state,
                             feed_dict={self.batch_size: grids.shape[0], self.sampled_grids: grids,
                                        self.current_position_map: curr_positions, self.previous_position_map: prev_positions, 
                                        })
        return Q

    def train_n(self, curr_state_tuple, action, target_Q, batch_size, lr, epsilon, iter, first_step):
        grids, curr_positions, prev_positions = curr_state_tuple
        loss, Q = self.sess.run([self.loss, self.predict],
                                   feed_dict={self.batch_size: batch_size, self.learning_rate: lr, self.sampled_grids: grids,
                                              self.current_position_map: curr_positions, self.previous_position_map: prev_positions,
                                              self.action: action, self.target: target_Q})
        if first_step:
            self.sess.run(self.train_first_step,
                                   feed_dict={self.batch_size: batch_size, self.learning_rate: lr, self.sampled_grids: grids,
                                              self.current_position_map: curr_positions, self.previous_position_map: prev_positions,
                                              self.action: action, self.target: target_Q})
            self.sess.run(self.save_state)
            self.sess.run(self.propagate_state,
                                    feed_dict={self.batch_size: batch_size, self.learning_rate: lr, self.sampled_grids: grids,
                                              self.current_position_map: curr_positions, self.previous_position_map: prev_positions,
                                              self.action: action, self.target: target_Q})
        else:
            self.sess.run(self.train_nn,
                                   feed_dict={self.batch_size: batch_size, self.learning_rate: lr, self.sampled_grids: grids,
                                              self.current_position_map: curr_positions, self.previous_position_map: prev_positions,
                                              self.action: action, self.target: target_Q})

        meanQ = np.mean(Q)
        maxQ = np.max(Q)
        # Log to tensorboard
        self.log_to_tensorboard(tag='Loss', group='drone_2D', value=np.linalg.norm(loss) / batch_size, index=iter)
        self.log_to_tensorboard(tag='Epsilon', group='drone_2D', value=epsilon, index=iter)
        # self.log_to_tensorboard(tag='Learning Rate', group='drone_2D', value=lr, index=iter)
        self.log_to_tensorboard(tag='MeanQ', group='drone_2D', value=meanQ, index=iter)
        self.log_to_tensorboard(tag='MaxQ', group='drone_2D', value=maxQ, index=iter)
        return np.array2string(Q), np.array2string(loss)

    def action_selection(self, curr_state_tuple, first_step):
        grids, curr_positions, prev_positions = curr_state_tuple
        if first_step: self.sess.run(self.restore_state)
        qvals = self.sess.run(self.predict,
                              feed_dict={self.batch_size: grids.shape[0],
                                         self.sampled_grids: grids, self.current_position_map: curr_positions,
                                         self.previous_position_map: prev_positions})

        # np.argmax gives me index into the flattened qvals array. np.unravel_index returns me the actual indexes of the original shape
        # print('model max action:', np.argmax(qvals))

        action = np.unravel_index(np.argmax(qvals), (52, 52))
        # print('converted action:', action)
        return np.array([action])

    def action_selection_non_repeat(self, curr_state_tuple, previous_actions, first_step):
        """Same as action selection, but extra filtering to prevent repeated actions"""
        grids, curr_positions, prev_positions = curr_state_tuple
        if first_step: self.sess.run(self.restore_state)
        qvals = self.sess.run(self.predict,
                              feed_dict={self.batch_size: grids.shape[0],
                                         self.sampled_grids: grids, self.current_position_map: curr_positions,
                                         self.previous_position_map: prev_positions})
        self.sess.run(self.propagate_state, 
                                feed_dict={self.batch_size: grids.shape[0],
                                         self.sampled_grids: grids, self.current_position_map: curr_positions,
                                         self.previous_position_map: prev_positions})
        while True:
            action_idx = np.unravel_index(np.argmax(qvals), (52, 52))
            repeated_action = action_idx in previous_actions

            if repeated_action:
                qvals[np.argmax(qvals)] = -np.inf
            else:
                return np.array([action_idx])

    def log_to_tensorboard(self, tag, group, value, index):
        summary = tf.Summary()
        tag = group + '/' + tag
        summary.value.add(tag=tag, simple_value=value)
        self.stat_writer.add_summary(summary, index)

    def save_network(self, iter=''):
        save_path = self.save_weights_path + '/drone_2D_' + iter
        self.saver.save(self.sess, save_path, write_meta_graph=False)
        print('Model Saved: ', save_path)

    def load_network(self, load_path):
        self.saver.restore(self.sess, load_path)
