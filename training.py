
#TODO:
#random data feed - validate
#make reusable MyModel for all slots
#delex, include request slot
#how to handle overall new words -xavie
#what to use : config file or FLAGS?
#rt calculation - reduce_sum or reduce_mean


import os
import sys
import argparse
import joblib
import time
import glob


import random
import json
from configparser import RawConfigParser
import codecs 

from nbt_modified import load_word_vectors, xavier_vector, load_woz_data, generate_data, generate_examples, extract_feature_vectors, delexicalise_utterance_values, print_slot_predictions, return_slot_predictions, print_belief_state_woz_requestables, print_belief_state_woz_informable, generate_examples_predict, generate_npa_examples

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow_addons as tfa

from model import RequestSlot_Model, NonRequestSlot_Model, Delex_Model, NIE_Model, NPA_Model, NLG_GRU_Model, P_Model, Q_Model
from evaluate import softmax_accuracy, request_accuracy

#tf.debugging.set_log_device_placement(True)

import numpy as np
np.set_printoptions(edgeitems=100000, linewidth=100000, precision=2, suppress=True)

FLAGS = None

epsilon = 1e-15

def test_utterance(model, utterances, word_vectors, dialogue_ontology, target_slot, slot_vectors, value_vectors, do_print=True):
    """
    Returns a list of belief states, to be weighted later. 
    """

    potential_values = dialogue_ontology[target_slot]

    if target_slot == "request":
        value_count = len(potential_values)
    else:
        value_count = len(potential_values) + 1

    # should be a list of features for each ngram supplied. 
    fv_tuples = extract_feature_vectors(utterances, word_vectors, use_asr=True)
    utterance_count = len(utterances)

    belief_state = np.zeros((value_count,), dtype="float32")

    # accumulators
    slot_values = []
    candidate_values = []
    delexicalised_features = []
    fv_full = []
    fv_sys_req = []
    fv_conf_slot = []
    fv_conf_val = []
    fv_masks = []
    features_previous_state = []

    for idx_hyp, extracted_fv in enumerate(fv_tuples):

        current_utterance = utterances[idx_hyp][0][0]

        prev_belief_state = utterances[idx_hyp][4]

        prev_belief_state_vector = np.zeros((value_count,), dtype="float32")
        
        if target_slot != "request":

            prev_value = prev_belief_state[target_slot]

            if prev_value == "none" or prev_value not in dialogue_ontology[target_slot]:
                prev_belief_state_vector[value_count-1] = 1
            else:
                prev_belief_state_vector[dialogue_ontology[target_slot].index(prev_value)] = 1

        features_previous_state.append(prev_belief_state_vector)

        (full_utt, masks, sys_req, conf_slot, conf_value) = extracted_fv 

        delex_vector = delexicalise_utterance_values(current_utterance, target_slot, dialogue_ontology[target_slot])

        fv_full.append(full_utt)
        delexicalised_features.append(delex_vector)
        fv_sys_req.append(sys_req)
        fv_conf_slot.append(conf_slot)
        fv_conf_val.append(conf_value)
        fv_masks.append(masks)

    slot_values = np.array(slot_values)
    candidate_values = np.array(candidate_values)
    delexicalised_features = np.array(delexicalised_features) # will be [batch_size, label_size, longest_utterance_length, vector_dimension]

    fv_sys_req = np.array(fv_sys_req)
    fv_conf_slot = np.array(fv_conf_slot)
    fv_conf_val = np.array(fv_conf_val)
    fv_masks = np.array(fv_masks)
    features_previous_state = np.array(features_previous_state)

    b_hat = model(fv_full, fv_sys_req, fv_conf_slot, fv_conf_val, delexicalised_features, features_previous_state, slot_vectors, value_vectors)

    #keep_prob, x_full, x_delex, \
    #requested_slots, system_act_confirm_slots, system_act_confirm_values, y_, y_past_state, accuracy, \
    #f_score, precision, recall, num_true_positives, \
    #num_positives, classified_positives, y, predictions, true_predictions, correct_prediction, \
    #true_positives, train_step, update_coefficient = model_variables

    #distribution, update_coefficient_load = sess.run([y, update_coefficient], feed_dict={x_full: fv_full, x_delex: delexicalised_features, \
    #                                  requested_slots: fv_sys_req, \
    #                                  system_act_confirm_slots: fv_conf_slot, y_past_state: features_previous_state, system_act_confirm_values: fv_conf_val, \
    #                                  keep_prob: 1.0})

    if target_slot == "request":
      distribution = tf.nn.sigmoid(b_hat)
    else:
      distribution = tf.math.softmax(b_hat)

    current_start_idx = 0
    list_of_belief_states = []

    for idx in range(0, utterance_count):
        current_distribution = distribution[idx, :]
        list_of_belief_states.append(current_distribution)

    if do_print:
      print_slot_predictions(distribution[0].numpy(), potential_values, target_slot, threshold=0.1)

    if len(list_of_belief_states) == 1:
        return [list_of_belief_states[0]]

    return list_of_belief_states

class WarmingSchedule(tf.optimizers.schedules.ExponentialDecay):
  def __init__(self, 
		warmup_steps,
		initial_learning_rate,
		minimal_learning_rate,
		decay_steps,
		decay_rate=0.99,
		staircase=False):
    super(WarmingSchedule, self).__init__(initial_learning_rate, decay_steps, decay_rate=decay_rate, staircase=staircase)

    self.warmup_steps = warmup_steps
    self.initial_learning_rate = initial_learning_rate
    self.minimal_learning_rate = minimal_learning_rate

  def __call__(self, step):
    rate = tf.case([(tf.equal(self.warmup_steps, 0), lambda: self.initial_learning_rate)], lambda: tf.minimum(self.initial_learning_rate*(1/self.warmup_steps)*tf.cast(step+1, tf.float32), super(WarmingSchedule, self).__call__(step)))
    return tf.case([(tf.less_equal(step, self.warmup_steps), lambda: rate)], lambda: tf.maximum(rate, self.minimal_learning_rate))

class Common(tf.Module):
  #   B = batch size (number of sequences)
  #   M = sentense length
  #   d - word vector size
  #   h - hidden size
  #   p = num slot/value pairs
  #   S = munber of slots
  #   v = values in slot
  #   a - num dialog actions
  #   V - vocabulary size
  def __init__(self):
    super(Common, self).__init__()

    self.config = RawConfigParser()
    self.config.read(FLAGS.config_file)
  
    self.dataset_name = self.config.get("model", "dataset_name")
  
    ontology_filepath = self.config.get("model", "ontology_filepath") 
    self.dialogue_ontology = json.load(codecs.open(ontology_filepath, "r", "utf-8"))
    self.dialogue_ontology = self.dialogue_ontology["informable"]

    self.word_vectors = {}
    word_vector_file = self.config.get("data", "word_vectors")
  
    lp = {}
    lp["english"] = u"en"
         
    self.language = self.config.get("model", "language")
    self.language_suffix = lp[self.language]
  
    self.ord_vectors = {}

    if os.path.exists("data/word_vectors.npy"):
      with open("data/word_vectors.npy", 'rb') as f:
        self.word_vectors["<unk>"] = np.load(f)
        self.word_vectors["</s>"] = np.load(f)
    else:
      self.word_vectors["<unk>"] = xavier_vector("<unk>")
      self.word_vectors["</s>"] = xavier_vector("</s>")
      with open("data/word_vectors.npy", 'wb') as f:
        np.save(f, word_vectors["<unk>"], allow_pickle=False)
        np.save(f, word_vectors["</s>"], allow_pickle=False)

    self.word_vectors = {**self.word_vectors, **load_word_vectors(word_vector_file, primary_language=self.language)}

    #initialize special tags
    self.word_vectors["tag-slot"] = xavier_vector("tag-slot")
    self.word_vectors["tag-value"] = xavier_vector("tag-value")
  
    slots = self.dialogue_ontology.keys()
  
    self.word_vector_size = random.choice(list(self.word_vectors.values())).shape[0]
  
    # a bit of hard-coding to make our lives easier. 
    if "price" in self.word_vectors and "range" in self.word_vectors:
      self.word_vectors["price range"] = self.word_vectors["price"] + self.word_vectors["range"]
    if "post" in self.word_vectors and "code" in self.word_vectors:
      self.word_vectors["postcode"] = self.word_vectors["post"] + self.word_vectors["code"]
    if "dont" in self.word_vectors and "care" in self.word_vectors:
      self.word_vectors["dontcare"] = self.word_vectors["dont"] + self.word_vectors["care"]
    if "addressess" in self.word_vectors:
      self.word_vectors["addressess"] = self.word_vectors["addresses"]
    if "dont" in self.word_vectors:
      self.word_vectors["don't"] = self.word_vectors["dont"]
  
    dontcare_value = "dontcare"
  
    #single word slot names are considered in the self.word_vectors!
    for slot_name in slots:
      if dontcare_value not in self.dialogue_ontology[slot_name] and slot_name != "request":
        self.dialogue_ontology[slot_name].append(dontcare_value)
        for value in self.dialogue_ontology[slot_name]:
          value = str(value)
          if " " not in value and value not in self.word_vectors:
  
            self.word_vectors[str(value)] = xavier_vector(str(value))
            #print("-- Generating word vector for:", value.encode("utf-8"), ":::", np.sum(self.word_vectors[value]))
  
      # add up multi-word word values to get their representation:
      for slot in list(self.dialogue_ontology.keys()):
        if " " in slot:
          slot = str(slot)
          self.word_vectors[slot] = np.zeros((self.word_vector_size,), dtype="float32")
          constituent_words = slot.split()
          for word in constituent_words:
            word = str(word)
            if word in self.word_vectors:
              self.word_vectors[slot] += self.word_vectors[word]
  
        for value in self.dialogue_ontology[slot]:
          if " " in value:
            value = str(value)
            self.word_vectors[value] = np.zeros((self.word_vector_size,), dtype="float32")
            constituent_words = value.split()
            for word in constituent_words:
              word = str(word)
              if word in self.word_vectors:
                self.word_vectors[value] += self.word_vectors[word]
  
    self.batches_per_epoch = int(self.config.get("train", "batches_per_epoch"))
    self.max_epoch = int(self.config.get("train", "max_epoch"))
    self.batch_size = int(self.config.get("train", "batch_size"))

class SlotModel(tf.Module):
  def __init__(self, common, target_slot, is_training):
    super(SlotModel, self).__init__()

    if target_slot == 'request':
      self.use_softmax = False
      self.model = RequestSlot_Model(
                  FLAGS.sentense_length,
                  len(common.dialogue_ontology[target_slot]),
                  word_vector_size=common.word_vector_size,
                  dropout_prob=FLAGS.dropout_prob,
                  initializer_range=1.0,
                  is_training=is_training)
    else:
      self.use_softmax = True
      self.model = NonRequestSlot_Model(
                  FLAGS.sentense_length,
                  len(common.dialogue_ontology[target_slot]),
                  word_vector_size=common.word_vector_size,
                  dropout_prob=FLAGS.dropout_prob,
                  initializer_range=1.0,
                  is_training=is_training)

    #0.0001 * 0.99 ^ (80000 / 1000) --> .000044
    #0.0001 * 0.99 ^ (40000 / 2000) --> .00008
    #0.0001 * 0.99 ^ (1000 / 5) --> .00001339
    #initial_learning_rate * decay_rate ^ (step / decay_steps)
  
    learning_rate_fn = WarmingSchedule(FLAGS.warmup_steps, FLAGS.learning_rate, FLAGS.minimal_rate, FLAGS.decay_steps, decay_rate=0.99, staircase=False)

    #learning_rate_fn = tf.optimizers.schedules.ExponentialDecay(FLAGS.learning_rate, FLAGS.decay_steps, decay_rate=0.99, staircase=False)

    self.checkpoint_prefix = os.path.join(FLAGS.output_dir, target_slot + "_ckpt")

    if is_training:
      self.optimizer = tf.optimizers.Adam(learning_rate_fn)
  
      self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, mymodel=self.model)
    else:
      self.checkpoint = tf.train.Checkpoint(mymodel=self.model)
  
    if len(glob.glob(self.checkpoint_prefix + '*')):
      if is_training:
        self.checkpoint.read(self.checkpoint_prefix)
      else:
        self.checkpoint.read(self.checkpoint_prefix).expect_partial()
    else:
      if not is_training:
        raise ValueError('Missing model for ', target_slot)

class DST(tf.Module):
  #   B = batch size (number of sequences)
  #   M = sentense length
  #   d - word vector size
  #   h - hidden size
  #   p = num slot/value pairs
  #   S = munber of slots
  #   v = values in slot
  #   a - num dialog actions
  #   V - vocabulary size
  def __init__(self, common, slot, is_training):
    super(DST, self).__init__()

    self.common = common
    
    if slot:
      self.slots = [slot]
    else:
      self.slots = list(self.common.dialogue_ontology.keys())      

    self.slot_models = {}
    for target_slot in self.slots:
      self.slot_models[target_slot] = SlotModel(self.common, target_slot, is_training)
    
  def evaluate_slot(self, model, target_slot, val_data, slot_vectors, value_vectors):
  
    (batch_xs_full, batch_masks, batch_sys_req, batch_sys_conf_slots, batch_sys_conf_values, batch_delex, batch_ys, batch_ys_prev) = val_data

    b_hat = model(batch_xs_full, batch_sys_req, batch_sys_conf_slots, batch_sys_conf_values, batch_delex, batch_ys_prev, slot_vectors, value_vectors)
  
    if target_slot == "request":
      y_hat = tf.nn.sigmoid(b_hat)
      accuracy = request_accuracy(y_hat, batch_ys)
    else:
      y_hat = tf.math.softmax(b_hat)
      accuracy = softmax_accuracy(y_hat, batch_ys)
  
    return accuracy

  def train(self):

    _, utterances_train2 = load_woz_data("data/" + self.common.dataset_name + "/" + self.common.dataset_name + "_train_" + self.common.language_suffix + ".json", self.common.language)
  
    utterance_count = len(utterances_train2)
  
    _, utterances_val2 = load_woz_data("data/" + self.common.dataset_name + "/" + self.common.dataset_name + "_validate_" + self.common.language_suffix + ".json", self.common.language)
  
    val_count = len(utterances_val2)
  
    utterances_train = utterances_train2 + utterances_val2[0:int(0.75 * val_count)]
    utterances_val = utterances_val2[int(0.75 * val_count):]
  
    #tf.print("\nTraining using:", dataset_name, " data - Utterance count:", utterance_count)
  
    # training feature vectors and positive and negative examples list. 
    #print("Generating data for training set:")
    feature_vectors, positive_examples, negative_examples = generate_data(utterances_train, self.common.word_vectors, self.common.dialogue_ontology)
        
    tf.print ("Generating data for validation set:")
    # same for validation (can pre-compute full representation, will be used after each epoch):
    fv_validation, positive_examples_validation, negative_examples_validation = \
                generate_data(utterances_val, self.common.word_vectors, self.common.dialogue_ontology)
    
    for target_slot in self.slots:
    
      val_data = generate_examples(target_slot, fv_validation, self.common.word_vectors, self.common.dialogue_ontology,
                                positive_examples_validation, negative_examples_validation)
    
      random_positive_count = int(self.common.batch_size / 2)
      random_negative_count = self.common.batch_size - random_positive_count
    
      if target_slot == 'request':
        slot_vectors = np.zeros((len(self.common.dialogue_ontology[target_slot]), 300), dtype="float32")
        value_vectors = np.zeros((len(self.common.dialogue_ontology[target_slot]), 300), dtype="float32")
      else:
        slot_vectors = np.zeros((len(self.common.dialogue_ontology[target_slot])+1, 300), dtype="float32") # +1 for None
        value_vectors = np.zeros((len(self.common.dialogue_ontology[target_slot])+1, 300), dtype="float32")
    
      for value_idx, value in enumerate(self.common.dialogue_ontology[target_slot]):
        slot_vectors[value_idx, :] = self.common.word_vectors[target_slot]
        value_vectors[value_idx, :] = self.common.word_vectors[value]
    
      epoch = int(self.slot_models[target_slot].optimizer.iterations / self.common.batches_per_epoch) + 1

      while epoch < self.common.max_epoch+1:
        for _ in range(self.common.batches_per_epoch):
          with tf.GradientTape() as tape:
            start_time=time.time()
    
            batch_data = generate_examples(target_slot, feature_vectors, self.common.word_vectors, self.common.dialogue_ontology,
                    positive_examples, negative_examples, random_positive_count, random_negative_count)
    
            (batch_xs_full, batch_masks, batch_sys_req, batch_sys_conf_slots, batch_sys_conf_values, batch_delex, batch_ys, batch_ys_prev) = batch_data
    
            #tf.print ("BATCH:", batch_xs_full.shape, batch_sys_req.shape, batch_sys_conf_slots.shape, batch_sys_conf_values.shape, batch_delex.shape, batch_ys.shape, batch_ys_prev.shape)
    
            b_hat = self.slot_models[target_slot].model(batch_xs_full, batch_sys_req, batch_sys_conf_slots, batch_sys_conf_values, batch_delex, batch_ys_prev, slot_vectors, value_vectors)
            if self.slot_models[target_slot].use_softmax:
              per_example_loss = tf.nn.softmax_cross_entropy_with_logits(labels=batch_ys, logits=b_hat, axis=-1, name=None)
              y_hat = tf.math.softmax(b_hat)
              train_accurarcy = softmax_accuracy(y_hat, batch_ys)
            else:
              y_hat = tf.nn.sigmoid(b_hat)
              per_example_loss = tf.reduce_sum(tf.square(y_hat - tf.stop_gradient(batch_ys)), axis=1)
              train_accurarcy = request_accuracy(y_hat, batch_ys)
    
            total_loss = tf.reduce_mean(per_example_loss)
    
            gradients = tape.gradient(total_loss, self.slot_models[target_slot].variables)
            if (FLAGS.clip_gradients > 0):
              gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.clip_gradients)
    
            self.slot_models[target_slot].optimizer.apply_gradients(zip(gradients, self.slot_models[target_slot].variables))
    
            tf.print("loss-" + target_slot.replace(' ', '_') + ":", self.slot_models[target_slot].optimizer.iterations, epoch, total_loss, train_accurarcy, self.slot_models[target_slot].optimizer.lr(self.slot_models[target_slot].optimizer.iterations), time.time() - start_time, output_stream=sys.stderr, summarize=-1)
    
            if self.slot_models[target_slot].optimizer.iterations % FLAGS.save_batches == 0 or self.slot_models[target_slot].optimizer.iterations == 1:
              self.slot_models[target_slot].checkpoint.write(file_prefix=self.slot_models[target_slot].checkpoint_prefix)
    
            if self.slot_models[target_slot].optimizer.iterations % 50 == 0:

              slot_model = SlotModel(self.common, target_slot, False)

              eval_accuracy = self.evaluate_slot(slot_model.model, target_slot, val_data, slot_vectors, value_vectors)
    
              tf.print("evaluation-" + target_slot.replace(' ', '_') + ":", self.slot_models[target_slot].optimizer.iterations, epoch, total_loss, eval_accuracy, self.slot_models[target_slot].optimizer.lr(self.slot_models[target_slot].optimizer.iterations), time.time() - start_time, output_stream=sys.stderr, summarize=-1)

            epoch = int(self.slot_models[target_slot].optimizer.iterations / self.common.batches_per_epoch) + 1

  def test(self, slot):
  
    _, utterances_val = load_woz_data("data/" + self.common.dataset_name + "/" + self.common.dataset_name + "_test_" + self.common.language_suffix + ".json", self.common.language)
    tf.print ("loaded test data: ", len(utterances_val))
  
    tf.print ("Generating data for validation set:")
    # same for validation (can pre-compute full representation, will be used after each epoch):
    fv_validation, positive_examples_validation, negative_examples_validation = \
                generate_data(utterances_val, self.common.word_vectors, self.common.dialogue_ontology)
  
    overall_accuracy = 1
    for target_slot in self.slots:

      val_data = generate_examples(target_slot, fv_validation, self.common.word_vectors, self.common.dialogue_ontology,
                                  positive_examples_validation, negative_examples_validation)
    
      if target_slot == "request":
                    
        slot_vectors = np.zeros((len(self.common.dialogue_ontology[target_slot]), 300), dtype="float32")
        value_vectors = np.zeros((len(self.common.dialogue_ontology[target_slot]), 300), dtype="float32")
    
        for value_idx, value in enumerate(self.common.dialogue_ontology[target_slot]):
          slot_vectors[value_idx, :] = self.common.word_vectors[target_slot]
          value_vectors[value_idx, :] = self.common.word_vectors[value]
      else:
                    
        slot_vectors = np.zeros((len(self.common.dialogue_ontology[target_slot])+1, 300), dtype="float32") # +1 for None
        value_vectors = np.zeros((len(self.common.dialogue_ontology[target_slot])+1, 300), dtype="float32")
    
        for value_idx, value in enumerate(self.common.dialogue_ontology[target_slot]):
          slot_vectors[value_idx, :] = self.common.word_vectors[target_slot]
          value_vectors[value_idx, :] = self.common.word_vectors[value]
    
      eval_accuracy = self.evaluate_slot(slot_models[target_slot], target_slot, val_data, slot_vectors, value_vectors)
  
      tf.print ("Accuracy for ", target_slot, ": ", eval_accuracy)
      
      overall_accuracy *= eval_accuracy
  
    tf.print ("Overall: ", overall_accuracy)

  def predict(self):
  
    dialogues, utterances_predict = load_woz_data("data/" + self.common.dataset_name + "/" + self.common.dataset_name + "_test_" + self.common.language_suffix + ".json", self.common.language)
    predict_count = len(utterances_predict)
  
    slot_vectors = {}
    value_vectors = {}
  
    request = 0
    inform = 0
    for target_slot in slots:

      if slot == "request":
        request + 1
                  
        slot_vectors[target_slot] = np.zeros((len(self.common.dialogue_ontology[target_slot]), 300), dtype="float32")
        value_vectors[target_slot] = np.zeros((len(self.common.dialogue_ontology[target_slot]), 300), dtype="float32")
  
        for value_idx, value in enumerate(self.common.dialogue_ontology[target_slot]):
          slot_vectors[target_slot][value_idx, :] = self.common.word_vectors[target_slot]
          value_vectors[target_slot][value_idx, :] = self.common.word_vectors[value]
      else:
                  
        inform += 1
        slot_vectors[target_slot] = np.zeros((len(self.common.dialogue_ontology[target_slot])+1, 300), dtype="float32") # +1 for None
        value_vectors[target_slot] = np.zeros((len(self.common.dialogue_ontology[target_slot])+1, 300), dtype="float32")
  
        for value_idx, value in enumerate(self.common.dialogue_ontology[target_slot]):
          slot_vectors[target_slot][value_idx, :] = self.common.word_vectors[target_slot]
          value_vectors[target_slot][value_idx, :] = self.common.word_vectors[value]
    tf.print ("#request #inform: ", request, inform)
    dialogue_count = len(dialogues)
    for idx in range(0, dialogue_count):
  
      prediction_dict = {}
      current_bs = {}
  
      #for each utterance
      for idx, trans_and_req_and_label_and_currlabel in enumerate(dialogues[idx]):
        (transcription_and_asr, req_slot, conf_slot, conf_value, label, prev_belief_state) = trans_and_req_and_label_and_currlabel
  
        for slot in slots:
          example = [(transcription_and_asr, req_slot, conf_slot, conf_value, prev_belief_state)] 
  
          tf.print ("True state for ", slot, ": ", label[slot])
          belief_states = test_utterance(slot_models[slot].model, example, self.common.word_vectors, self.common.dialogue_ontology, slot, slot_vectors[slot], value_vectors[slot], do_print=True)
          belief_state = belief_states[0]
  
          predicted_values = return_slot_predictions(belief_state.numpy(), self.common.dialogue_ontology[slot], slot, 0.5)
          tf.print ("belief_state for ", slot, ": ", predicted_values)
  
          if slot in "request":
            current_bs[slot] = print_belief_state_woz_requestables(self.common.dialogue_ontology[slot], belief_state, threshold=0.5)
          else:
            current_bs[slot] = print_belief_state_woz_informable(self.common.dialogue_ontology[slot], belief_state, threshold=0.01) # swap to 0.001 Nikola
         
        tf.print("belief state for utterance: ", current_bs)
  
  def predict_batch(self, feature_vectors, utterances_train, positive_indices):
  
    y_batch = tf.zeros([self.common.batch_size, 0], dtype=tf.float32)

    for target_slot in self.slots:

      #need : batch_xs_full, batch_masks, batch_sys_req, batch_sys_conf_slots, batch_sys_conf_values, batch_delex, batch_ys_prev
      #get data for slots from the same samples (indeces)
      batch_data = generate_examples_predict(target_slot, feature_vectors, self.common.word_vectors, self.common.dialogue_ontology, utterances_train, positive_indices)

      if target_slot == "request":
                    
        slot_vectors = np.zeros((len(self.common.dialogue_ontology[target_slot]), 300), dtype="float32")
        value_vectors = np.zeros((len(self.common.dialogue_ontology[target_slot]), 300), dtype="float32")
    
        for value_idx, value in enumerate(self.common.dialogue_ontology[target_slot]):
          slot_vectors[value_idx, :] = self.common.word_vectors[target_slot]
          value_vectors[value_idx, :] = self.common.word_vectors[value]
      else:
                    
        slot_vectors = np.zeros((len(self.common.dialogue_ontology[target_slot])+1, 300), dtype="float32") # +1 for None
        value_vectors = np.zeros((len(self.common.dialogue_ontology[target_slot])+1, 300), dtype="float32")
    
        for value_idx, value in enumerate(self.common.dialogue_ontology[target_slot]):
          slot_vectors[value_idx, :] = self.common.word_vectors[target_slot]
          value_vectors[value_idx, :] = self.common.word_vectors[value]
    
      (batch_xs_full, batch_masks, batch_sys_req, batch_sys_conf_slots, batch_sys_conf_values, batch_delex, batch_ys_prev) = batch_data

      b_hat = self.slot_models[target_slot].model(batch_xs_full, batch_sys_req, batch_sys_conf_slots, batch_sys_conf_values, batch_delex, batch_ys_prev, slot_vectors, value_vectors)
  
      if target_slot == "request":
        y_hat = tf.nn.sigmoid(b_hat)
      else:
        y_hat = tf.math.softmax(b_hat)

      tf.print (target_slot)
      tf.print ("y_batch shape", y_batch.shape)
      tf.print ("y_hat shape", y_hat.shape)

      y_batch = tf.concat([y_batch, y_hat], axis=1)

    return y_batch

class VIN(tf.Module):
  #   B = batch size (number of sequences)
  #   M = sentense length
  #   d - word vector size
  #   h - hidden size
  #   p = num slot/value pairs
  #   S = munber of slots
  #   v = values in slot
  #   a - num dialog actions
  #   V - vocabulary size
  def __init__(self, common):
    super(VIN, self).__init__()

    is_training = True

    self.common = common
    
    _, utterances_train2 = load_woz_data("data/" + self.common.dataset_name + "/" + self.common.dataset_name + "_train_" + self.common.language_suffix + ".json", self.common.language)
  
    utterance_count = len(utterances_train2)
  
    _, utterances_val2 = load_woz_data("data/" + self.common.dataset_name + "/" + self.common.dataset_name + "_validate_" + self.common.language_suffix + ".json", self.common.language)
  
    val_count = len(utterances_val2)
  
    self.utterances_train = utterances_train2 + utterances_val2[0:int(0.75 * val_count)]
    utterances_val = utterances_val2[int(0.75 * val_count):]
  
    #tf.print("\nTraining using:", dataset_name, " data - Utterance count:", utterance_count)
  
    # training feature vectors and positive and negative examples list. 
    #print("Generating data for training set:")
    #feature_vectors, positive_examples, negative_examples = generate_data(utterances_train, self.common.word_vectors, self.common.dialogue_ontology)
    self.feature_vectors = extract_feature_vectors(self.utterances_train, self.common.word_vectors)
      
    #tf.print ("Generating data for validation set:")
    # same for validation (can pre-compute full representation, will be used after each epoch):
    #fv_validation, positive_examples_validation, negative_examples_validation = \
    #              generate_data(utterances_val, self.common.word_vectors, self.common.dialogue_ontology)
  
    #val_data = generate_examples(target_slot, fv_validation, self.common.word_vectors, self.common.dialogue_ontology,
    #                              positive_examples_validation, negative_examples_validation)
  
    self.delex_model = Delex_Model(FLAGS.delex_threshold)

    self.dkl_lambda = FLAGS.dkl_lambda
  
    #self.group_normalization_1 = tfa.layers.GroupNormalization(groups=1, axis=-1)

    self.word_vectors_array=np.array(list(self.common.word_vectors.values()), dtype='float32')

    self.nie_model = NIE_Model(
      batch_size=self.common.batch_size,
      hidden_size=FLAGS.hidden_size,
      sentense_length=FLAGS.sentense_length,
      dropout_prob=FLAGS.dropout_prob,
      initializer_range=1.0,
      is_training=is_training)
  
    self.npa_model = NPA_Model(
      self.common.batch_size,
      hidden_size=FLAGS.hidden_size,
      num_pairs=FLAGS.num_pairs,
      num_actions=FLAGS.num_actions,
      dropout_prob=FLAGS.dropout_prob,
      initializer_range=1.0,
      is_training=is_training)

    self.nlg_model = NLG_GRU_Model(
      self.common.batch_size,
      num_pairs=FLAGS.num_pairs,
      num_actions=FLAGS.num_actions,
      hidden_size=FLAGS.hidden_size,
      sentense_length=FLAGS.sentense_length,
      word_vectors=self.word_vectors_array,
      dropout_prob=FLAGS.dropout_prob,
      initializer_range=1.0,
      is_training=is_training)

    self.q_model = Q_Model(
      self.dkl_lambda,
      self.common.batch_size,
      num_pairs=FLAGS.num_pairs,
      num_slots=FLAGS.num_slots,
      num_actions=FLAGS.num_actions,
      hidden_size=FLAGS.hidden_size,
      sentense_length=FLAGS.sentense_length,
      vocabulary_size=self.word_vectors_array.shape[0],
      dropout_prob=FLAGS.dropout_prob,
      initializer_range=1.0,
      is_training=is_training)

    self.p_model = P_Model(
      self.nie_model,
      self.npa_model,
      self.nlg_model,
      self.common.batch_size,
      num_pairs=FLAGS.num_pairs,
      num_slots=FLAGS.num_slots,
      num_actions=FLAGS.num_actions,
      hidden_size=FLAGS.hidden_size,
      sentense_length=FLAGS.sentense_length,
      vocabulary_size=self.word_vectors_array.shape[0],
      dropout_prob=FLAGS.dropout_prob,
      initializer_range=1.0,
      is_training=is_training)

    #0.0001 * 0.99 ^ (80000 / 1000) --> .000044
    #0.0001 * 0.99 ^ (40000 / 2000) --> .00008
    #0.0001 * 0.99 ^ (1000 / 5) --> .00001339
    #initial_learning_rate * decay_rate ^ (step / decay_steps)
  
    learning_rate_fn = WarmingSchedule(FLAGS.warmup_steps, FLAGS.learning_rate, FLAGS.minimal_rate, FLAGS.decay_steps, decay_rate=0.99, staircase=False)
    #learning_rate_fn = tf.optimizers.schedules.ExponentialDecay(FLAGS.learning_rate, FLAGS.decay_steps, decay_rate=0.99, staircase=False)
  
    self.checkpoint_prefix = os.path.join(FLAGS.output_dir, "vin" + "_ckpt")

    if is_training:
      self.optimizer = tf.optimizers.Adam(learning_rate_fn)
  
      self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, nie_model=self.nie_model, npa_model=self.npa_model, nlg_model=self.nlg_model, p_model=self.p_model, q_model=self.q_model)
    else:
      self.checkpoint = tf.train.Checkpoint(nie_model=self.nie_model, npa_model=self.npa_model, nlg_model=self.nlg_model)
  
    if len(glob.glob(self.checkpoint_prefix + '*')):
      if is_training:
        self.checkpoint.read(self.checkpoint_prefix)
      else:
        self.checkpoint.read(self.checkpoint_prefix).expect_partial()
    else:
      if not is_training:
        raise ValueError('Missing model NPA/NIE')

  def print_utterance(self, Dt_index, prefix="", sep="|"):
    wv = list(self.common.word_vectors.keys())
    for sentence in Dt_index:
      s = prefix
      for i in sentence:
        s += wv[i]
        s += sep
      tf.print (s)

  def train(self):

    random_positive_count = int(self.common.batch_size / 2)
    random_negative_count = self.common.batch_size - random_positive_count
  
    slots = []
    value_vectors = {}
    slot_vectors = np.zeros((len(self.common.dialogue_ontology)-1, self.common.word_vector_size), dtype="float32")
    slot_idx = 0
    informable_pairs = 0
    for slot in self.common.dialogue_ontology.keys():
      if slot != "request":
        slots.append(slot) 
        slot_vectors[slot_idx, :] = self.common.word_vectors[slot]
        value_vectors[slot] = np.zeros((len(self.common.dialogue_ontology[slot]), self.common.word_vector_size), dtype="float32")
        for value_idx, value in enumerate(self.common.dialogue_ontology[slot]):
          value_vectors[slot][value_idx, :] = self.common.word_vectors[value]
          informable_pairs += 1
        slot_idx += 1
    tf.print ("informable_pairs: ", informable_pairs) 
    epoch = int(self.optimizer.iterations / self.common.batches_per_epoch) + 1

    dst = DST(self.common, FLAGS.slot, False)
    while epoch < self.common.max_epoch+1:
      for _ in range(self.common.batches_per_epoch):
        with tf.GradientTape() as tape:
          start_time=time.time()
  
          positive_indices = np.random.choice(len(self.utterances_train), self.common.batch_size)

          #input featues, DA is the same regardless target_lot: if to send batch of utterences? 
          bt = dst.predict_batch(self.feature_vectors, self.utterances_train, positive_indices)

          tf.print ("DST: ", bt.shape)

          batch_data = generate_npa_examples(self.feature_vectors, self.common.word_vectors, self.common.dialogue_ontology, self.utterances_train, positive_indices)

          (batch_xs_full, batch_masks, batch_sys_req, batch_sys_conf_slots, batch_sys_conf_values, batch_delex, batch_target, batch_information_richness) = batch_data
  
          #tf.print ("BATCH:", batch_xs_full.shape, batch_masks.shape, batch_sys_req.shape, batch_sys_conf_slots.shape, batch_sys_conf_values.shape, batch_delex.shape, batch_ys.shape, batch_ys_prev.shape, batch_target)
  
          delex_batch = []
          #(B, M, d) --> (M, B, d)
          for token_batch in np.transpose(batch_xs_full, [1, 0, 2]):
            #(B, d), (S, d) --> (B)  slot id or null
            delex_values = np.zeros((self.common.batch_size, self.common.word_vector_size), dtype="float32")
            best_slots = self.delex_model.best_slots(token_batch, batch_sys_req, batch_sys_conf_slots, batch_sys_conf_values, batch_delex, slot_vectors)
            for token_idx, token in enumerate(token_batch):
              if (best_slots[token_idx] < len(slots)):
                slot = slots[best_slots[token_idx]]
                best_value = self.delex_model.best_values(token, batch_sys_req, batch_sys_conf_slots, batch_sys_conf_values, batch_delex, value_vectors[slot])
                if (best_value < len(value_vectors[slot])):
                  delex_values[token_idx, :] = slot_vectors[best_slots[token_idx]] + value_vectors[slot][best_value] 
                else:
                  delex_values[token_idx, :] = token
            delex_batch.append(delex_values)
  
          delex_batch = np.stack(delex_batch, axis=1)

          tf.print ("delex: ", delex_batch.shape)

          zt, Dt, Dt_index, Dt_loss, action_distribution, action_prob = self.p_model(delex_batch, batch_masks, batch_target, batch_information_richness, bt)

          self.print_utterance(batch_target[:1,1:], "TAR:")
          self.print_utterance(Dt_index[:1,:], "OUT:")

          q_distribution, q_prob_onehot, q_prob, br = self.q_model(bt, batch_information_richness, zt, Dt)

          Q = tf.stop_gradient(q_distribution)

          #(b, M) --> (b)
          per_example_p_loss = -tf.reduce_sum(tf.math.log(Dt_loss + epsilon), axis=-1)
          #((b, a) * ((b, a) - log(b, a))) --> (b)
          per_example_KL_loss = self.dkl_lambda*tf.reduce_sum(Q*(tf.math.log(Q + epsilon) - tf.math.log(action_distribution + epsilon)), axis=-1)
  
          total_p_loss = tf.reduce_mean(per_example_p_loss)
          total_KL_loss = tf.reduce_mean(per_example_KL_loss)
          total_loss = total_p_loss + total_KL_loss
  
          tf.print ("Dt_loss, q, action_prob shapes: ", Dt_loss.shape, q_prob.shape, action_prob.shape)

          #(b, M), (b, a) --> (b) , (b) --> (b)
          r = tf.stop_gradient(tf.reduce_sum(tf.math.log(Dt_loss + epsilon), axis=-1) - tf.reduce_sum(self.dkl_lambda*(q_distribution*(tf.math.log(q_distribution + epsilon) - tf.math.log(action_distribution + epsilon))), axis=-1))

          reward = tf.stop_gradient(r - br)
          base = br

          #(b), (b, a) --> (b), (b) --> (b)
          per_example_q_loss = -reward*tf.math.log(q_prob + epsilon)
          total_q_loss = tf.reduce_mean(per_example_q_loss)

          per_example_base_loss = tf.reduce_mean(tf.square(r - base))
          base_loss = tf.reduce_mean(per_example_base_loss)

          all_loss = total_p_loss + total_KL_loss + total_q_loss + base_loss

          all_vars = self.p_model.variables + self.q_model.variables
  
          gradients = tape.gradient(all_loss, all_vars)
          if (FLAGS.clip_gradients > 0):
            gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.clip_gradients)
  
          self.optimizer.apply_gradients(zip(gradients, all_vars))
  
          tf.print("loss:", self.optimizer.iterations, epoch, total_p_loss, total_KL_loss, total_loss, tf.reduce_mean(r), tf.reduce_mean(tf.math.log(q_prob + epsilon)), total_q_loss, base_loss, tf.reduce_mean(reward), self.optimizer.lr(self.optimizer.iterations), time.time() - start_time, output_stream=sys.stderr, summarize=-1)
  
          if self.optimizer.iterations % FLAGS.save_batches == 0 or self.optimizer.iterations == 1:
            self.checkpoint.write(file_prefix=self.checkpoint_prefix)
  
          epoch = int(self.optimizer.iterations / self.common.batches_per_epoch) + 1
  
def main():  
  common = Common()
  if FLAGS.action == 'TRAIN_DST':
    DST(common, True).train()
  if FLAGS.action == 'TRAIN_VIN':
    VIN(common).train()
  elif FLAGS.action == 'TEST_DST':
    with tf.device('/cpu:0'):
      DST(common, False).test(FLAGS.slot)
  elif FLAGS.action == 'PREDICT_DST':
    with tf.device('/cpu:0'):
      DST(common, False).predict(FLAGS.slot)

if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument('--config_file', type=str, default='config/predict.tfrecords',
            help='Predict file location in google storage.')
  parser.add_argument('--sentense_length', type=int, default=40,
            help='Max sentense lenght.')
  parser.add_argument('--hidden_size', type=int, default=50,
            help='Embedding/hidden size.')
  parser.add_argument('--delex_threshold', type=float, default=0.9,
            help='Threshold to replace sentence slot/values embeddings with ontology slot/values.')
  parser.add_argument('--num_pairs', type=int, default=112,
            help='Number of inform and request slot/values pairs including dontcare and none.')
  parser.add_argument('--num_slots', type=int, default=4,
            help='Number of inform slot/values pairs including dontcare and none.')
  parser.add_argument('--num_actions', type=int, default=70,
            help='These are dialog actions as in dialogs.')
  parser.add_argument('--dkl_lambda', type=float, default=0.1,
            help='Hyperparameter to regularize KL divergence.')

  parser.add_argument('--output_dir', type=str, default='checkpoints',
            help='Model directrory in google storage.')
  parser.add_argument('--save_batches', type=int, default=100,
            help='Save every N batches.')
  parser.add_argument('--feedforward_size', type=int, default=64,
            help='Last non-linearity layer in the transformer.')
  parser.add_argument('--num_hidden_layers', type=int, default=2,
            help='One self-attention block only.')
  parser.add_argument('--num_attention_heads', type=int, default=4,
            help='number of attention heads in transformer.')
  parser.add_argument('--dropout_prob', type=float, default=0.2,
            help='This used for all dropouts.')
  parser.add_argument('--learning_rate', type=float, default=1e-3,
            help='Optimizer initial learning rate.')
  parser.add_argument('--minimal_rate', type=float, default=1e-4,
            help='Optimizer minimal learning rate.')
  parser.add_argument('--decay_steps', type=int, default=500000,
            help='Exponential decay parameter.')
  parser.add_argument('--warmup_steps', type=int, default=0,
            help='Learning rate grow from zero to initial rate during this time.')
  parser.add_argument('--logging', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'],
            help='Enable excessive variables screen outputs.')
  parser.add_argument('--batch_size', type=int, default=32,
            help='Batch size.')
  parser.add_argument('--clip_gradients', type=float, default=0.2,
            help='Clip gradients to deal with explosive gradients.')
  parser.add_argument('--action', default='PREDICT', choices=['TRAIN_DST', 'TRAIN_VIN', 'TEST_DST', 'PREDICT_DST'],
            help='An action to execure.')
  parser.add_argument('--slot', choices=['request', 'food', 'price range', 'area'],
            help='Slot to train.')

  FLAGS, unparsed = parser.parse_known_args()

  tf.print ("Running with parameters: {}".format(FLAGS))

  main()
