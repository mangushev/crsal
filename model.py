
#TODO:
#fine tune dropout
#use biLSTM for NIE instead og biGRU
#vocabulary size is 500 after preprocessing in LIDM opposite to 65k
#need <<UKN>> token for all unknown to stabilize vocabulary size
#delex should be in sample preprocessing
#use LSTM for language model instead of GRU
#how to hook NRN?
#use biLSTM to generate d for input into inference network instead of biGRU
#target masks

import sys

import tensorflow as tf
import tensorflow_addons as tfa

import math

import numpy as np

epsilon = 1e-15

def dropout(input, dropout_prob):
  if dropout_prob == 0.0:
    return input
  return tf.nn.dropout(input, dropout_prob)

class Dense(tf.Module):
  def __init__(self, input_size, output_size, activation=None, stddev=1.0, name=''):
    super(Dense, self).__init__()
    self.w = tf.Variable(
      tf.random.truncated_normal([input_size, output_size], stddev=stddev), name=name + '_w')
    self.b = tf.Variable(tf.zeros([output_size]), name=name+'_b')
    self.activation = activation
    self.input_size = input_size
    self.output_size = output_size
  def __call__(self, x):
    input_shape = x.shape

    if len(input_shape) != 2 and len(input_shape) != 3:
      raise ValueError("input shape rank {} shuld be 2 or 3".format(len(input_shape)))

    if len(input_shape) == 3:
      #if self.input_size != input_shape[2]:
      #  raise ValueError("input size do not match {} {} {}".format(self.input_size, input_shape[2], input_shape))
      x = tf.reshape(x, [-1, self.input_size])
    else:
      x = x
      #if self.input_size != input_shape[1]:
      #  raise ValueError("input size do not match {} {}".format(self.input_size, input_shape[1]))

    y = tf.matmul(x, self.w) + self.b
    if (self.activation is not None):
      y = self.activation(y)

    if len(input_shape) == 3:
      return tf.reshape(y, [-1, input_shape[1], self.output_size])
#reshape required? see rensorflow matmul
    return y

class Conv1d_layer(tf.Module):
  def __init__(self, filter_width, in_channels, out_channels, padding='SAME', activation=None, stddev=1.0, name=''):
    super(Conv1d_layer, self).__init__()
    self.filter_width = filter_width
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.padding = padding
    self.activation = activation

    self.f = tf.Variable(tf.random.truncated_normal([self.filter_width, self.in_channels, self.out_channels], stddev=stddev), name=name + '_filter')
    #self.f = tf.Variable(tf.ones([self.filter_width, self.in_channels, self.out_channels]), name=name + '_filter')
    #self.b = tf.Variable(tf.zeros([self.out_channels]), name=name+'_b')
    self.b = tf.Variable(tf.constant(0.1, shape=[self.out_channels]), name=name+'_b')
  def __call__(self, x):
    conv = tf.nn.conv1d(x, self.f, [1], padding=self.padding, name="conv")
    conv_bias = tf.nn.bias_add(conv, self.b, name='conv_bias')

    if self.activation:
      return self.activation(conv_bias, name='conv_bias_relu')
    return conv_bias

class GRU_Encoder(tf.Module):
  def __init__(self, hidden_size, sequence_length, batch_size, dropout=0.1, is_training=True):
    super(GRU_Encoder, self).__init__()

    self.gru_cell = tf.keras.layers.GRUCell(hidden_size, activation='tanh', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', dropout=dropout, recurrent_dropout=0.0)

    self.hidden_size = hidden_size
    self.sequence_length = sequence_length
    self.batch_size = batch_size
    self.is_training = is_training

  def __call__(self, x):

    #[B, h, f] -> [h, B, f]
    step_inputs = tf.transpose(x, [1, 0, 2])

    step = tf.constant(0)
    output_ta = tf.TensorArray(size=self.sequence_length, dtype=tf.float32, clear_after_read=False)
    initial_state = tf.zeros((self.batch_size, self.hidden_size), dtype=tf.float32, name='state')

    def cond(step, output_ta, state):
      return tf.less(step, self.sequence_length)

    def body(step, output_ta, state):

      input = tf.slice(step_inputs, [step, 0, 0], [1, -1, -1], name='slice')
      input_one = tf.squeeze(input, axis=0, name='squeeze')
      output, state = self.gru_cell(input_one, state, training=self.is_training)

      output_ta = output_ta.write(step, output, name='ta_w')

      return (step + 1, output_ta, state)

    _, output_ta_final, state = tf.while_loop(cond, body, [step, output_ta, [initial_state]], name='gru_loop')

    time_gru_output = output_ta_final.stack(name='stack_ta')

    #[w, B, h] -> [B, w, h]
    return tf.transpose(time_gru_output, [1, 0, 2]), state

class biGRU_Encoder(tf.Module):
  def __init__(self, hidden_size, sequence_length, batch_size, dropout=0.1, is_training=True):
    super(biGRU_Encoder, self).__init__()

    self.cell_forward = tf.keras.layers.GRUCell(hidden_size, activation='tanh', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', dropout=dropout, recurrent_dropout=0.0, name="cell_forward") 
    self.cell_backward = tf.keras.layers.GRUCell(hidden_size, activation='tanh', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', dropout=dropout, recurrent_dropout=0.0, name="cell_backward") 

    self.hidden_size = hidden_size
    self.sequence_length = sequence_length
    self.batch_size = batch_size
    self.is_training = is_training

  def __call__(self, x, masks):

    masks = tf.cast(masks, 'bool')

    #[B, h, f] -> [h, B, f]
    step_inputs = tf.transpose(x, [1, 0, 2])

    step = tf.constant(0)
    forward_output_ta = tf.TensorArray(size=self.sequence_length, dtype=tf.float32, clear_after_read=False, element_shape=[self.batch_size, self.hidden_size])
    backward_output_ta = tf.TensorArray(size=self.sequence_length, dtype=tf.float32, clear_after_read=False, element_shape=[self.batch_size, self.hidden_size])

    forward_last_output = tf.zeros((self.batch_size, self.hidden_size), dtype=tf.float32, name='forward_output')
    backward_last_output = tf.zeros((self.batch_size, self.hidden_size), dtype=tf.float32, name='backward_output')

    forward_last_state = tf.zeros((self.batch_size, self.hidden_size), dtype=tf.float32, name='forward_state')
    backward_last_state = tf.zeros((self.batch_size, self.hidden_size), dtype=tf.float32, name='backward_state')

    def cond(step, forward_output_ta, backward_output_ta, forward_last_output, backward_last_output, forward_last_state, backward_last_state):
      return tf.less(step, self.sequence_length)

    def body(step, forward_output_ta, backward_output_ta, forward_last_output, backward_last_output, forward_last_state, backward_last_state):

      forward_input = tf.slice(step_inputs, [step, 0, 0], [1, -1, -1], name='forward_slice')
      forward_input_one = tf.squeeze(forward_input, axis=0, name='squeeze_forward')
      forward_output, forward_state = self.cell_forward(forward_input_one, forward_last_state, training=self.is_training)

      #(b) , (b, h), (b, h)
      forward_output = tf.where(masks[:, step:step+1], forward_output, forward_last_output[0])
      forward_state = tf.where(masks[:, step:step+1], forward_state, forward_last_state[0])

      backward_step = tf.add_n([self.sequence_length, -step, -1], name="backward_step")
      backward_input = tf.slice(step_inputs, [backward_step, 0, 0], [1, -1, -1], name='backward_slice')
      backward_input_one = tf.squeeze(backward_input, axis=0, name='sqeeze_backward')
      backward_output, backward_state = self.cell_backward(backward_input_one, backward_last_state, training=self.is_training)

      #(b) , (b, h), (b, h)
      backward_output = tf.where(masks[:, backward_step:backward_step+1], backward_output, backward_last_output[0])
      backward_state = tf.where(masks[:, backward_step:backward_step+1], backward_state, backward_last_state[0])
      
      forward_output_ta = forward_output_ta.write(step, forward_output, name='forward_ta_w')
      backward_output_ta = backward_output_ta.write(backward_step, backward_output, name='backward_ta_w')

      return (step + 1, forward_output_ta, backward_output_ta, forward_output, backward_output, forward_state, backward_state)

    _, forward_output_ta_final, backward_output_ta_final, _, _, forward_state_final, backward_state_final = tf.while_loop(cond, body, [step, forward_output_ta, backward_output_ta, forward_last_output, backward_last_output, forward_last_state, backward_last_state], name='rnn_loop')

    forward_projections = forward_output_ta_final.stack(name='stack_forward_ta')
    backward_projections = backward_output_ta_final.stack(name='stack_backward_ta')

    #[w, B, h] -> [B, w, h]
    return tf.transpose(forward_projections, [1, 0, 2]), tf.transpose(backward_projections, [1, 0, 2]), forward_state_final, backward_state_final

class GRU_Decoder(tf.Module):
  def __init__(self, hidden_size, sequence_length, batch_size, word_vectors, dropout=0.1, is_training=True, initializer_range=1.0):
    super(GRU_Decoder, self).__init__()

    self.gru_cell = tf.keras.layers.GRUCell(hidden_size, activation='tanh', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', dropout=dropout, recurrent_dropout=0.0)

    self.hidden_size = hidden_size
    self.sequence_length = sequence_length
    self.batch_size = batch_size
    self.word_vectors = word_vectors
    self.is_training = is_training

    self.V_layer = Dense(hidden_size,
                word_vectors.shape[0],
                activation=tf.nn.softmax,
                stddev=initializer_range,
                name='V_layer')

    self.Wdh = tf.Variable(tf.random.truncated_normal([word_vectors.shape[1], hidden_size]), name='Wih')

  def __call__(self, inputs, labels, ot_h):

    #(b, M-1) -> (M-1, b)
    inputs_tr = tf.transpose(inputs, [1, 0])
    labels_tr = tf.transpose(labels, [1, 0])

    step = tf.constant(0)
    loss_ta = tf.TensorArray(size=self.sequence_length, dtype=tf.float32, clear_after_read=False)
    output_ta = tf.TensorArray(size=self.sequence_length, dtype=tf.int64, clear_after_read=False)
    initial_state = tf.zeros([self.batch_size, self.hidden_size], dtype=tf.float32, name='state')

    def cond(step, loss_ta, output_ta, state):
      return tf.less(step, self.sequence_length-1)

    def body(step, loss_ta, output_ta, state):

      #(M-1, b) --> (b) --> (b, V) 
      input_onehot = tf.one_hot(inputs_tr[step], self.word_vectors.shape[0])

      #(b, V), (V, d) --> (b, d)
      input_d = tf.matmul(input_onehot, self.word_vectors)
      
      #(b, d), (d, h) --> (b, h)
      #sigmoid to normalize into [0,1], ot_h is tanh
      input_h = tf.nn.sigmoid(tf.matmul(input_d, self.Wdh))

      output, state = self.gru_cell(input_h + ot_h, state, training=self.is_training)
 
      #(b, h) --> (b, V)
      output_probs = self.V_layer(output)

      #(M-1, b) --> (b) --> (b, V) 
      labels_onehot = tf.one_hot(labels_tr[step], self.word_vectors.shape[0])

      #(b, V), (b, V) --> (b, V) --> (b)
      loss_ta = loss_ta.write(step, tf.reduce_sum(labels_onehot * output_probs, axis=-1))

      #(b, V) --> (b)
      output_ta = output_ta.write(step, tf.argmax(output_probs, -1))

      return (step + 1, loss_ta, output_ta, state)

    _, loss_ta_final, output_ta_final, _ = tf.while_loop(cond, body, [step, loss_ta, output_ta, initial_state], name='gru_loop')

    loss = loss_ta_final.stack()
    output = output_ta_final.stack()

    #loss: (M-1, b) -> (b, M-1); output: (M-1, b) --> (b, M-1)
    return tf.transpose(loss, [1, 0]), tf.transpose(output, [1, 0])

class RequestSlot_Model(tf.Module):
  #   B = batch size (number of sequences)
  #   M = sentense length
  #   d - word vector size
  #   h - hidden size
  #   p = num slot/value pairs
  #   S = munber of slots
  #   v = values in slot
  #   a - num dialog actions
  #   V - vocabulary size
  def __init__(self, sentense_size,
                     n_values,
                     word_vector_size,
                     activation_fn=tf.nn.relu,
                     dropout_prob=0.5,
                     initializer_range=1.0,
                     is_training=False):
    super(RequestSlot_Model, self).__init__()

    if is_training == False:
      dropout_prob = 0.0   

    self.dropout_prob = dropout_prob

    self.n_values = n_values

    self.conv_1n = Conv1d_layer(1, word_vector_size, word_vector_size, padding='VALID', stddev=0.1, activation=activation_fn)
    self.conv_2n = Conv1d_layer(2, word_vector_size, word_vector_size, padding='VALID', stddev=0.1, activation=activation_fn)
    self.conv_3n = Conv1d_layer(3, word_vector_size, word_vector_size, padding='VALID', stddev=0.1, activation=activation_fn)

    self.c_layer = Dense(word_vector_size,
                word_vector_size,
                activation=tf.nn.sigmoid,
                stddev=initializer_range,
                name='c_layer')

    self.sem_layer_1 = Dense(word_vector_size,
		100,
		activation=tf.nn.sigmoid,
		stddev=initializer_range,
		name='sem_layer_1')

    self.sem_layer_2 = Dense(100,
		1,
		activation=None,
		stddev=initializer_range,
		name='sem_layer_2')

  def __call__(self, utterence_representation, sys_req, sys_conf_slots, sys_conf_values, utterance_representations_delex, batch_ys_prev, slot_vectors, slot_values):

    #(v, d) + (v, d)) --> (v, d)
    #return self.c_layer(slot_vectors + slot_values)
    #(v, d),(d, d)) --> (v, d)
    candidates = self.c_layer(slot_values)

    #tf.print ("utterence_representation", utterence_representation.shape)

    #(B, M, d) --> (B, d)
    G = tf.math.reduce_max(self.conv_1n(utterence_representation), axis=1, keepdims=False)
    G = G + tf.math.reduce_max(self.conv_2n(utterence_representation), axis=1, keepdims=False)
    G = G + tf.math.reduce_max(self.conv_3n(utterence_representation), axis=1, keepdims=False)

    #semantic interaction using multiply
    #(B, d) * (v, d) --> (B, 1, d) * (1, v, d) --> (B, v, d)
    semantic_interaction = tf.expand_dims(G, 1) * tf.expand_dims(candidates[:self.n_values, :], 0)

    #applying semantic weight and sigmoid
    #(B,v,d) --> (B,v,h)
    sem_output = self.sem_layer_1(semantic_interaction)
    sem_output = dropout(sem_output, self.dropout_prob)
    #(B,v,h) --> (B,v,1)
    sem_output = self.sem_layer_2(sem_output)
    #(B,v,1) --> (B,v)
    sem_output = tf.squeeze(sem_output, axis=-1)

    #tf.print ("semantic interaction", sem_output.shape)

    y_presoftmax = sem_output

    #tf.print ("y_presoftmax", y_presoftmax.shape)
    #tf.print ("utterance_representations_delex", utterance_representations_delex.shape)

    #in addition flag any values found in the sentense
    #y_presoftmax = y_presoftmax + utterance_representations_delex

    #b1 = tf.math.sigmoid(y_presoftmax) # for request slot nothing is carried over

    return y_presoftmax

class NonRequestSlot_Model(tf.Module):
  #   B = batch size (number of sequences)
  #   M = sentense length
  #   d - word vector size
  #   h - hidden size
  #   p = num slot/value pairs
  #   S = munber of slots
  #   v = values in slot
  #   a - num dialog actions
  #   V - vocabulary size
  def __init__(self, sentense_size,
                     n_values,
                     word_vector_size,
                     activation_fn=tf.nn.relu,
                     dropout_prob=0.5,
                     initializer_range=1.0,
                     is_training=False):
    super(NonRequestSlot_Model, self).__init__()

    if is_training == False:
      dropout_prob = 0.0   

    self.dropout_prob = dropout_prob

    self.n_values = n_values

    self.conv_1n = Conv1d_layer(1, word_vector_size, word_vector_size, padding='VALID', stddev=0.1, activation=activation_fn)
    self.conv_2n = Conv1d_layer(2, word_vector_size, word_vector_size, padding='VALID', stddev=0.1, activation=activation_fn)
    self.conv_3n = Conv1d_layer(3, word_vector_size, word_vector_size, padding='VALID', stddev=0.1, activation=activation_fn)

    self.c_layer = Dense(word_vector_size,
                word_vector_size,
                activation=tf.nn.sigmoid,
                stddev=initializer_range,
                name='c_layer')

    self.sem_layer_1 = Dense(word_vector_size,
		100,
		activation=tf.nn.sigmoid,
		stddev=initializer_range,
		name='sem_layer_1')

    self.sem_layer_2 = Dense(100,
		1,
		activation=None,
		stddev=initializer_range,
		name='sem_layer_2')

    self.req_layer_1 = Dense(word_vector_size,
		100,
		activation=tf.nn.sigmoid,
		stddev=initializer_range,
		name='req_layer_1')

    self.req_layer_2 = Dense(100,
		n_values,
		activation=None,
		stddev=initializer_range,
		name='req_layer_2')

    self.conf_layer_1 = Dense(word_vector_size,
		100,
		activation=tf.nn.sigmoid,
		stddev=initializer_range,
		name='conf_layer_1')

    self.conf_layer_2 = Dense(100,
		1,
		activation=None,
		stddev=initializer_range,
		name='conf_layer_2')

    self.W1_past = tf.Variable(tf.random.truncated_normal([n_values+1, n_values+1]), name='W1_past')
    self.W2_past = tf.Variable(tf.random.truncated_normal([n_values+1, n_values+1]), name='W2_past')
    self.W1_current = tf.Variable(tf.random.truncated_normal([n_values+1, n_values+1]), name='W1_current')
    self.W2_current = tf.Variable(tf.random.truncated_normal([n_values+1, n_values+1]), name='W2_current')

  def __call__(self, utterence_representation, sys_req, sys_conf_slots, sys_conf_values, utterance_representations_delex, batch_ys_prev, slot_vectors, slot_values):

    #(v, d) + (v, d)) --> (v, d)
    #return self.c_layer(slot_vectors + slot_values)
    #(v, d),(d, d)) --> (v, d)
    candidates = self.c_layer(slot_values)

    #tf.print ("utterence_representation", utterence_representation.shape)

    #(B, M, d) --> (B, d)
    G = tf.math.reduce_max(self.conv_1n(utterence_representation), axis=1, keepdims=False)
    G = G + tf.math.reduce_max(self.conv_2n(utterence_representation), axis=1, keepdims=False)
    G = G + tf.math.reduce_max(self.conv_3n(utterence_representation), axis=1, keepdims=False)

    #semantic interaction using multiply
    #(B, d) * (v, d) --> (B, 1, d) * (1, v, d) --> (B, v, d)
    semantic_interaction = tf.expand_dims(G, 1) * tf.expand_dims(candidates[:self.n_values, :], 0)

    #applying semantic weight and sigmoid
    #(B,v,d) --> (B,v,h)
    sem_output = self.sem_layer_1(semantic_interaction)
    sem_output = dropout(sem_output, self.dropout_prob)
    #(B,v,h) --> (B,v,1)
    sem_output = self.sem_layer_2(sem_output)
    #(B,v,1) --> (B,v)
    sem_output = tf.squeeze(sem_output, axis=-1)

    #tf.print ("semantic interaction", sem_output.shape)

    #context request interation

    #see if current processing slot is in system request using multiply
    #(B, d) * (1, d) --> (B, d)
    product_sysreq = sys_req * tf.expand_dims(slot_vectors[0, :], 0)
    #(B, d) --> (B)
    product_sysreq = tf.reduce_mean(product_sysreq, 1)
    
    #request interaction with utterence
    #(B, 1) * (B, d) --> (B, d)
    request_decision = tf.expand_dims(product_sysreq, 1) * G

    #applying request weight and sigmoid
    #(B,d) --> (B,h)
    req_output = self.req_layer_1(request_decision)
    req_output = dropout(req_output, self.dropout_prob)
    #(B,h) --> (B,v)
    req_output = self.req_layer_2(req_output)

    #tf.print ("request interaction", req_output.shape)
    
    #context confirmation interation
    
    #mean((B, d) * (1, d)) --> (B)
    product_conf_slot = tf.reduce_mean(sys_conf_slots * tf.expand_dims(slot_vectors[0, :], 0), 1)
    #mean((B, 1, d) * (1, v, d)) --> (B, V)
    product_conf_values = tf.reduce_mean(tf.expand_dims(sys_conf_values, 1) * tf.expand_dims(slot_values[:self.n_values, :], 0), -1)
    #(B, 1) * (B, v) --> (B, v)
    product_conf = tf.expand_dims(product_conf_slot, -1) * product_conf_values

    full_ones = tf.ones(tf.shape(product_conf))
    #(B, V) --> (B, V)
    product_conf_ones = tf.cast(tf.equal(product_conf, full_ones), "float32")

    #conf interaction with utterence
    #(B, v, 1) * (B, 1, d) --> (B, v, d)
    conf_decision = tf.expand_dims(product_conf_ones, -1) * tf.expand_dims(G, 1)

    #applying conf weight and sigmoid
    #(B,v,d) --> (B,v,h)
    conf_output = self.conf_layer_1(conf_decision)
    conf_output = dropout(conf_output, self.dropout_prob)
    #(B,v,h) --> (B,v,1)
    conf_output = self.conf_layer_2(conf_output)
    #(B,v,1) --> (B,v)
    conf_output = tf.squeeze(conf_output, axis=-1)

    #tf.print ("conf interaction", conf_output.shape)

    #add None and sum up all contibutions
    append_zeros_none = tf.zeros([tf.shape(sem_output)[0], 1])
    sem_output = tf.concat([sem_output, append_zeros_none], 1)

    append_zeros = tf.zeros([tf.shape(sem_output)[0], 1])
    req_output = tf.concat([req_output, append_zeros], 1)
    conf_output = tf.concat([conf_output, append_zeros], 1)

    y_presoftmax = sem_output + conf_output + req_output

    #tf.print ("y_presoftmax", y_presoftmax.shape)
    #tf.print ("utterance_representations_delex", utterance_representations_delex.shape)

    #in addition flag any values found in the sentense
    #y_presoftmax = y_presoftmax + utterance_representations_delex

    #(B, v), (v, v)--> (B, v)
    b1 = tf.matmul(batch_ys_prev, self.W1_past) + tf.matmul(y_presoftmax, self.W1_current)
    b2 = tf.matmul(batch_ys_prev, self.W2_past) + tf.matmul(y_presoftmax, self.W2_current)
    b12 = tf.concat([tf.expand_dims(b1, -1), tf.expand_dims(b2, -1)], -1)
    b = tf.math.softmax(b12)
    #b = b1 <= b2
    decision_b = tf.cast(tf.math.greater(b[:,:,0], b[:,:,1]), "float32")

    #softmax is done with cross_entropy
    return b12[:,:,0]

class Delex_Model(tf.Module):
  #   B = batch size (number of sequences)
  #   M = sentense length
  #   d - word vector size
  #   h - hidden size
  #   p = num slot/value pairs
  #   S = munber of slots
  #   v = values in slot
  #   a - num dialog actions
  #   V - vocabulary size
  def __init__(self, delex_threshold=0.9):
    super(Delex_Model, self).__init__()

    self.delex_threshold=delex_threshold

  def best_slots(self, utterence_representation, sys_req, sys_conf_slots, sys_conf_values, utterance_representations_delex, slot_vectors):

    #(B, d) . (S, d) --> (B, d) . (d, S) --> (B, S)
    product_slot = tf.nn.sigmoid(tf.matmul(utterence_representation, slot_vectors, transpose_b=True))

    slot_indices = tf.argmax(product_slot, -1, output_type=tf.dtypes.int32)
    slot_probs = tf.gather(product_slot, slot_indices, batch_dims=1, axis=-1)

    return tf.maximum(slot_indices, tf.cast(tf.cast(tf.less(slot_probs, self.delex_threshold), 'float32') * slot_vectors.shape[0], 'int32'))

  def best_values(self, utterence_representation, sys_req, sys_conf_slots, sys_conf_values, utterance_representations_delex, value_vectors):

    #(1, d) . (d, v) --> (1, v)
    product_value = tf.nn.sigmoid(tf.matmul(tf.expand_dims(utterence_representation, 0), value_vectors, transpose_b=True))
            
    value_indices = tf.argmax(product_value, -1, output_type=tf.dtypes.int32)
    value_probs = tf.gather(product_value, value_indices, batch_dims=1, axis=-1)

    return tf.squeeze(tf.maximum(value_indices, tf.cast(tf.cast(tf.less(value_probs, self.delex_threshold), 'float32') * value_vectors.shape[0], 'int32')))

class NIE_Model(tf.Module):
  #   B = batch size (number of sequences)
  #   M = sentense length
  #   d - word vector size
  #   h - hidden size
  #   p = num slot/value pairs
  #   S = munber of slots
  #   v = values in slot
  #   a - num dialog actions
  #   V - vocabulary size
  def __init__(self, batch_size,
		hidden_size,
		sentense_length,
		dropout_prob=0.1,
		initializer_range=1.0,
		is_training=False):
    super(NIE_Model, self).__init__()

    if is_training == False:
      dropout_prob = 0.0   

    self.dropout_prob = dropout_prob

    self.encoder_layer = biGRU_Encoder(
      hidden_size,
      sentense_length,
      batch_size,
      dropout=self.dropout_prob,
      is_training=is_training)

  def __call__(self, input, masks):
    #(b, M, d) --> (b, 2h)
    _, _, forward_state, backward_state = self.encoder_layer(input, masks)

    return tf.concat([forward_state, backward_state], axis=-1)

class NPA_Model(tf.Module):
  #   B = batch size (number of sequences)
  #   M = sentense length
  #   d - word vector size
  #   h - hidden size
  #   p = num slot/value pairs
  #   S = munber of slots
  #   v = values in slot
  #   a - num dialog actions
  #   V - vocabulary size
  def __init__(self, batch_size,
		hidden_size,
                num_pairs,
                num_actions,
		dropout_prob=0.1,
		initializer_range=1.0,
		is_training=False):
    super(NPA_Model, self).__init__()

    if is_training == False:
      dropout_prob = 0.0   

    self.num_actions = num_actions

    self.dropout_prob = dropout_prob

    self.layer_1 = Dense(hidden_size,
                hidden_size,
                activation=tf.math.tanh,
                stddev=initializer_range,
                name='layer_1')

    self.layer_2 = Dense(hidden_size,
                num_actions,
                activation=tf.nn.softmax,
                stddev=initializer_range,
                name='layer_2')
  
  def __call__(self, input):
    #(b, h) --> (b, a)
    action_distribution = self.layer_2(self.layer_1(input))

    #(b, a) --> (b)
    sampled_action = tf.squeeze(tf.random.categorical(tf.math.log(action_distribution + epsilon), 1))

    #(b) --> (b, a)
    #action_mask = tf.cast(tf.one_hot(sampled_action, self.num_actions), 'bool')
    action_mask = tf.one_hot(sampled_action, self.num_actions)

    #(b, a), (b, a) --> (b)
    action_prob_onehot = action_distribution * action_mask
    action_prob = tf.reduce_sum(action_prob_onehot, axis=-1)
    #action_prob = tf.squeeze(tf.ragged.boolean_mask(action_distribution, action_mask), axis=-1)

    return action_distribution, action_prob_onehot, action_prob

class NLG_GRU_Model(tf.Module):
  #   B = batch size (number of sequences)
  #   M = sentense length
  #   d - word vector size
  #   h - hidden size
  #   p = num slot/value pairs
  #   S = munber of slots
  #   v = values in slot
  #   a - num dialog actions
  #   V - vocabulary size
  def __init__(self, batch_size,
                num_pairs,
                num_actions,
		hidden_size,
		sentense_length,
                word_vectors,
		dropout_prob=0.1,
		initializer_range=1.0,
		is_training=False):
    super(NLG_GRU_Model, self).__init__()

    if is_training == False:
      dropout_prob = 0.0   

    self.dropout_prob = dropout_prob
    self.batch_size = batch_size
    self.word_vectors = word_vectors

    self.group_normalization_1 = tfa.layers.GroupNormalization(groups=1, axis=-1)

    self.Wgate = tf.Variable(tf.random.truncated_normal([num_actions, hidden_size]), name='Wgate')
    self.gate_layer = Dense(hidden_size,
                hidden_size,
                activation=tf.nn.sigmoid,
                stddev=initializer_range,
                name='gate_layer')

    self.W5 = tf.Variable(tf.random.truncated_normal([num_actions, hidden_size]), name='W5')

    self.initial_state = tf.zeros([batch_size, hidden_size], dtype=tf.float32, name='state')
    
    self.Woh = tf.Variable(tf.random.truncated_normal([2*hidden_size, hidden_size]), name='Woh')

    self.generator = GRU_Decoder(hidden_size, 
                sentense_length, 
                batch_size,
                word_vectors,
                dropout=dropout_prob,
                is_training=is_training,
                initializer_range=initializer_range)

  def __call__(self, s, a, inputs, labels):

    #ACTION - THIS MUST BE EMBEDDED
    #(b, a), (b, a) * (b, p+2h).(p+2h, a) --> (b, a), (b, a) --> (b, 2a) 

    gate = self.gate_layer(tf.matmul(a, self.Wgate))

    #gate is sigmoid folters s which is tanh. all of it contat with tanh of emb of a
    ot = tf.concat([tf.math.tanh(tf.matmul(a, self.W5)), gate * s], axis=-1)

    #(b, 2h) --> (b, h)
    #this is tanh of matmul of ot
    ot_h = tf.math.tanh(tf.matmul(ot, self.Woh))

    #(b, M-1), (b, M-1), (b, h) --> (b, M-1), (b, M-1)
    loss, output = self.generator(inputs, labels, ot_h)

    #(b, M-1) --> (b, M-1, V)
    output_onehot = tf.one_hot(output, self.word_vectors.shape[0])

    #(b, M-1, V), (V, d) --> (b, M-1, d)
    D = tf.matmul(output_onehot, self.word_vectors)

    return loss, output, D

class P_Model(tf.Module):
  #   B = batch size (number of sequences)
  #   M = sentense length
  #   d - word vector size
  #   h - hidden size
  #   p = num slot/value pairs
  #   S = munber of slots
  #   v = values in slot
  #   a - num dialog actions
  #   V - vocabulary size
  def __init__(self, 
                nie_model,
                npa_model,
                nlg_model,
                batch_size,
                num_pairs,
                num_slots,
                num_actions,
		hidden_size,
		sentense_length,
                vocabulary_size,
		dropout_prob=0.1,
		initializer_range=1.0,
		is_training=False):
    super(P_Model, self).__init__()

    if is_training == False:
      dropout_prob = 0.0   

    self.nie_model = nie_model
    self.npa_model = npa_model
    self.nlg_model = nlg_model
    self.num_actions = num_actions
    self.dropout_prob = dropout_prob
    self.batch_size = batch_size
    self.vocabulary_size = vocabulary_size
    self.hidden_size = hidden_size
    self.sentense_length = sentense_length

    self.Wq = tf.Variable(tf.random.truncated_normal([num_slots, hidden_size]), name='Wq')
    self.Wb = tf.Variable(tf.random.truncated_normal([num_pairs, hidden_size]), name='Wb')
    self.Wz = tf.Variable(tf.random.truncated_normal([2*hidden_size, hidden_size]), name='Wz')

  def __call__(self, delex_batch, batch_masks, batch_target, batch_information_richness, bt):

    #(b, M, d), (b, M) --> (b, 2h)
    zt = self.nie_model(delex_batch, batch_masks)
  
    tf.print ("NIE: ", zt.shape)
  
    #V - value count for all inform values (request as well?)
    #(b, p), (b, 2h) --> (b, p+2h)
    #s = q concat bt concat zt
    #s = tf.concat([bt, zt], axis=1)
    #ACTUALLY DOT EACH INTO RIGHT VECTOR SIZE AND SUM-UP OPPOSITE TO CONCAT
    #(b, p), (b, 2h) --> (b, h), (b, h) --> (b, h)
    #tf.print (batch_information_richness)
    s = tf.nn.tanh(tf.matmul(tf.cast(batch_information_richness, tf.float32), self.Wq)+tf.matmul(bt, self.Wb)+tf.matmul(zt, self.Wz))

    tf.print ("S state: ", s.shape)

    #(b, 2p) --> (b, a), (b, a), (b)
    #(b, h) --> (b, a), (b, a), (b)
    action_distribution, action_prob_onehot, action_prob = self.npa_model(s)
  
    tf.print ("NPA: ", action_distribution.shape, action_prob.shape)
  
    #(b, 2p), (b, a) --> (b, M), (b, M), (b, M, d)
    #(b, h), (b, a) --> (b, M), (b, M), (b, M, d)
    Dt_loss, Dt_index, Dt = self.nlg_model(s, action_prob_onehot, batch_target[:, :-1], batch_target[:, 1:])

    tf.print ("NLG: ", Dt.shape, Dt_index.shape, Dt_loss.shape)

    return zt, Dt, Dt_index, Dt_loss, action_distribution, action_prob 

class Q_Model(tf.Module):
  #   B = batch size (number of sequences)
  #   M = sentense length
  #   d - word vector size
  #   h - hidden size
  #   p = num slot/value pairs
  #   s = num slots
  #   S = munber of slots
  #   v = values in slot
  #   a - num dialog actions
  #   V - vocabulary size
  def __init__(self, 
                dkl_lambda,
                batch_size,
                num_pairs,
                num_slots,
                num_actions,
		hidden_size,
		sentense_length,
                vocabulary_size,
		dropout_prob=0.1,
		initializer_range=1.0,
		is_training=False):
    super(Q_Model, self).__init__()

    if is_training == False:
      dropout_prob = 0.0   

    self.dkl_lambda = dkl_lambda
    self.num_actions = num_actions
    self.dropout_prob = dropout_prob
    self.batch_size = batch_size
    self.vocabulary_size = vocabulary_size
    self.hidden_size = hidden_size
    self.sentense_length = sentense_length

    self.d_maker = biGRU_Encoder(
      hidden_size,
      sentense_length,
      batch_size,
      dropout=dropout_prob,
      is_training=is_training)

    self.Wq = tf.Variable(tf.random.truncated_normal([num_slots, hidden_size]), name='Wq')
    self.Wb = tf.Variable(tf.random.truncated_normal([num_pairs, hidden_size]), name='Wb')
    self.Wz = tf.Variable(tf.random.truncated_normal([hidden_size*2, hidden_size]), name='Wz')
    self.Wd = tf.Variable(tf.random.truncated_normal([hidden_size*2, hidden_size]), name='Wd')
    self.beta_y = tf.Variable(tf.zeros([hidden_size]), name='beta_y')
    self.We = tf.Variable(tf.random.truncated_normal([hidden_size, num_actions]), name='We')
    self.br = tf.Variable(0., name='br')

    self.masks = tf.constant(tf.ones((self.batch_size, self.sentense_length), dtype=tf.float32, name='masks'))

  def __call__(self, bt, batch_information_richness, zt, D):

    #Variational Inference Network (used for training only)

    #(b, M, d), (b, M) -->(b, d), (b, d)
    _, _, forward_state, backward_state = self.d_maker(D, self.masks)

    dt = tf.concat([forward_state, backward_state], axis=-1)

    tf.print ("bt, qt, zt, dt: ", bt.shape, batch_information_richness.shape, zt.shape, dt.shape)
    #(b, p), (b, s), (b, 2h), (b, 2h), (h) --> (b, h), (b, h), (b, h), (b, h), (h) --> (b, h)
    e_all = tf.nn.sigmoid(tf.matmul(bt, self.Wb) + tf.matmul(tf.cast(batch_information_richness, tf.float32), self.Wq) + tf.matmul(zt, self.Wz) + tf.matmul(dt, self.Wd) + self.beta_y)

    #(b, h) . (h, a) --> (b, a)
    q_distribution = tf.nn.softmax(tf.matmul(e_all, self.We))

    #(b, a) --> (b)
    sampled_q = tf.squeeze(tf.random.categorical(tf.math.log(q_distribution + epsilon), 1))

    #(b) --> (b, a)
    q_mask = tf.one_hot(sampled_q, self.num_actions)
    #tf.print (sampled_q, summarize=-1)
    #(b, a), (b, a) --> (b, a)
    q_prob_onehot = q_distribution * q_mask
    q_prob = tf.reduce_sum(q_prob_onehot, axis=-1)

    return q_distribution, q_prob_onehot, q_prob, self.br
