
#TODO:
#random data feed - validate
#make reusable MyModel for all slots
#delex rarely gets into .9 only postcode term
#how to handle overall new words -xavie
#reduc vocabulary by removing unused words
#optimize br
#what to use : config file or FLAGS?
#Q negative sign?
#rt calculation - reduce_sum or reduce_mean


import os
import sys
import argparse
import joblib
import time
import glob
from copy import deepcopy
from nltk.translate.bleu_score import sentence_bleu


import random
import json
from configparser import RawConfigParser
import codecs 

from nbt_modified import load_word_vectors, xavier_vector, load_woz_data, generate_data, generate_examples, extract_feature_vectors, delexicalise_utterance_values, print_slot_predictions, return_slot_predictions, print_belief_state_woz_requestables, print_belief_state_woz_informable, generate_examples_predict, generate_npa_examples, process_turn_hyp

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow_addons as tfa

from model import RequestSlot_Model, NonRequestSlot_Model, NIE_Model, NPA_Model, NLG_GRU_Model, P_Model

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

        #keep the same curr , prev believe state
        prev_belief_state = utterances[idx_hyp][5]

        prev_belief_state_vector = np.zeros((value_count,), dtype="float32")
        
        if target_slot != "request":

            prev_value = prev_belief_state[target_slot]

            if prev_value == "none" or prev_value not in dialogue_ontology[target_slot]:
                prev_belief_state_vector[value_count-1] = 1
            else:
                prev_belief_state_vector[dialogue_ontology[target_slot].index(prev_value)] = 1

        features_previous_state.append(prev_belief_state_vector)

        (full_utt, masks, sys_req, conf_slot, conf_value, _, _, _, _, _, _) = extracted_fv 

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

    word_vector_file = self.config.get("data", "word_vectors")
  
    restaurant_db_filepath = self.config.get("model", "restaurant_db")
    self.restaurant_db = json.load(codecs.open(restaurant_db_filepath, "r", "utf-8"))
    self.restaurant_db = [r["name"] for r in self.restaurant_db]

    self.num_informable_pairs = FLAGS.num_informable_pairs

    lp = {}
    lp["english"] = u"en"
         
    self.language = self.config.get("model", "language")
    self.language_suffix = lp[self.language]
  
    original_word_vectors = load_word_vectors(word_vector_file, primary_language=self.language)

    if os.path.exists("data/unknown_words.npy"):
      unk = np.load("data/unknown_words.npy", allow_pickle=True).item()
      self.word_vectors = {**unk, **original_word_vectors}
    else:
      self.word_vectors = deepcopy(original_word_vectors)

      self.word_vectors["<unk>"] = xavier_vector("<unk>")
      self.word_vectors["</s>"] = xavier_vector("</s>")

    self.word_vector_size = random.choice(list(self.word_vectors.values())).shape[0]

    #initialize special tags
    #self.word_vectors["tag-slot"] = xavier_vector("tag-slot")
    #self.word_vectors["tag-value"] = xavier_vector("tag-value")
  
    slots = self.dialogue_ontology.keys()
  
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
            tf.print("-- Generating word vector for:", value.encode("utf-8"), ":::", np.sum(self.word_vectors[value]))
  
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

    _, utterances_train2 = load_woz_data("data/" + self.dataset_name + "/" + self.dataset_name + "_train_" + self.language_suffix + ".json", self.language, self.num_informable_pairs, self.restaurant_db)
  
    utterance_count = len(utterances_train2)
  
    _, utterances_val2 = load_woz_data("data/" + self.dataset_name + "/" + self.dataset_name + "_validate_" + self.language_suffix + ".json", self.language, self.num_informable_pairs, self.restaurant_db)
  
    val_count = len(utterances_val2)
  
    self.utterances_train = utterances_train2 + utterances_val2[0:int(0.75 * val_count)]
    self.utterances_val = utterances_val2[int(0.75 * val_count):]

    _, self.utterances_test = load_woz_data("data/" + self.dataset_name + "/" + self.dataset_name + "_test_" + self.language_suffix + ".json", self.language, self.num_informable_pairs, self.restaurant_db)

    utterances = self.utterances_train + self.utterances_val + self.utterances_test
  
    #tf.print("\nTraining using:", dataset_name, " data - Utterance count:", utterance_count)
  
    use_asr = True
    for idx, utterance in enumerate(utterances):

      if use_asr:
        full_asr = utterances[idx][0][1] # just use ASR
      else:
        full_asr = [(utterances[idx][0][0], 1.0)] # else create (transcription, 1.0)

      asr_count = 1

      for (c_example, asr_coeff) in full_asr[0:asr_count]:

        if c_example != "":
          words_utterance = process_turn_hyp(c_example, "en")
          words_utterance = words_utterance.split()

          for word_idx, word in enumerate(words_utterance):

            if word not in self.word_vectors:
              self.word_vectors[word] = xavier_vector(word)
              tf.print("== Over Utterance: Generating random word vector for", word.encode('utf-8'), ":::", np.sum(self.word_vectors[word]))
                        
      for idx, word in enumerate(utterances[idx][6].split()):

        if word not in self.word_vectors:
          self.word_vectors[word] = xavier_vector(word)
          tf.print("== Over Utterance: Generating random word vector for", word.encode('utf-8'), ":::", np.sum(self.word_vectors[word]))

    unknown_words = {}
    for w, v in self.word_vectors.items():
      if w not in original_word_vectors:
        unknown_words[w] = v

    #tf.print ("Unknown words", unknown_words.keys(), len(unknown_words.keys()))

    if len(unknown_words.keys()) > 0:
      np.save("data/unknown_words.npy", unknown_words, allow_pickle=True)
  
    self.word_list = list(self.word_vectors.keys())

class SlotModel(tf.Module):
  def __init__(self, common, target_slot, is_training):
    super(SlotModel, self).__init__()

    if target_slot == 'request':
      self.use_softmax = False
      self.model = RequestSlot_Model(
                  FLAGS.sentense_length,
                  len(common.dialogue_ontology[target_slot]),
                  word_vector_size=common.word_vector_size,
                  hidden_size=FLAGS.dst_hidden_size,
                  dropout_prob=FLAGS.dropout_prob,
                  initializer_range=1.0,
                  is_training=is_training)
    else:
      self.use_softmax = True
      self.model = NonRequestSlot_Model(
                  FLAGS.sentense_length,
                  len(common.dialogue_ontology[target_slot]),
                  word_vector_size=common.word_vector_size,
                  hidden_size=FLAGS.dst_hidden_size,
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

    feature_vectors, positive_examples, negative_examples = generate_data(self.common.utterances_train, self.common.word_vectors, self.common.dialogue_ontology)
        
    fv_validation, positive_examples_validation, negative_examples_validation = \
                generate_data(self.common.utterances_val, self.common.word_vectors, self.common.dialogue_ontology)
    
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
              y_hat = tf.math.softmax(b_hat)
              per_example_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(batch_ys), logits=b_hat, axis=-1)
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

  def test(self):
  
    fv_validation, positive_examples_validation, negative_examples_validation = \
                generate_data(self.common.utterances_test, self.common.word_vectors, self.common.dialogue_ontology)
  
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
    
      eval_accuracy = self.evaluate_slot(self.slot_models[target_slot].model, target_slot, val_data, slot_vectors, value_vectors)
  
      tf.print ("Accuracy for ", target_slot, ": ", eval_accuracy)
      
      overall_accuracy *= eval_accuracy
  
    tf.print ("Overall: ", overall_accuracy)

  def predict(self):
  
    dialogues, utterances_predict = load_woz_data("data/" + self.common.dataset_name + "/" + self.common.dataset_name + "_test_" + self.common.language_suffix + ".json", self.common.language, self.common.num_informable_pairs, self.common.restaurant_db)
    predict_count = len(utterances_predict)
  
    slot_vectors = {}
    value_vectors = {}
  
    request = 0
    inform = 0
    for target_slot in self.slots:

      if target_slot == "request":
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
        (transcription_and_asr, req_slot, conf_slot, conf_value, label, prev_belief_state, current_system_transcript, current_information_richness, offered, list_system_requests, list_user_requests) = trans_and_req_and_label_and_currlabel
  
        for slot in self.slots:
          example = [(transcription_and_asr, req_slot, conf_slot, conf_value, None, prev_belief_state, current_system_transcript, current_information_richness, offered, list_system_requests, list_user_requests)] 
  
          tf.print ("True state for ", slot, ": ", label[slot])
          belief_states = test_utterance(self.slot_models[slot].model, example, self.common.word_vectors, self.common.dialogue_ontology, slot, slot_vectors[slot], value_vectors[slot], do_print=True)
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
    
    # training feature vectors and positive and negative examples list. 
    #print("Generating data for training set:")
    #feature_vectors, positive_examples, negative_examples = generate_data(utterances_train, self.common.word_vectors, self.common.dialogue_ontology)
    self.feature_vectors = extract_feature_vectors(self.common.utterances_train, self.common.word_vectors)
      
    #tf.print ("Generating data for validation set:")
    # same for validation (can pre-compute full representation, will be used after each epoch):
    #fv_validation, positive_examples_validation, negative_examples_validation = \
    #              generate_data(utterances_val, self.common.word_vectors, self.common.dialogue_ontology)
  
    #val_data = generate_examples(target_slot, fv_validation, self.common.word_vectors, self.common.dialogue_ontology,
    #                              positive_examples_validation, negative_examples_validation)
  
    self.word_vectors_array=np.array(list(self.common.word_vectors.values()), dtype='float32')

    self.nie_model = NIE_Model(
      self.common.batch_size,
      num_pairs=FLAGS.num_pairs,
      num_slots=FLAGS.num_slots,
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

    self.p_model = P_Model(
      FLAGS.dkl_lambda, 
      FLAGS.eta,
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
  
    self.learning_rate_fn = WarmingSchedule(FLAGS.warmup_steps, FLAGS.learning_rate, FLAGS.minimal_rate, FLAGS.decay_steps, decay_rate=0.99, staircase=False)
    #learning_rate_fn = tf.optimizers.schedules.ExponentialDecay(FLAGS.learning_rate, FLAGS.decay_steps, decay_rate=0.99, staircase=False)
  
    self.checkpoint_prefix = os.path.join(FLAGS.output_dir, "vin" + "_ckpt")

    if is_training:
      self.optimizer = tf.optimizers.Adam(self.learning_rate_fn)
      self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, p_model=self.p_model)
      #self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, npa_model=self.npa_model, nlg_model=self.nlg_model, p_model=self.p_model)
    else:
      self.checkpoint = tf.train.Checkpoint(nie_model=self.nie_model, npa_model=self.npa_model, nlg_model=self.nlg_model)
      #self.checkpoint = tf.train.Checkpoint(npa_model=self.npa_model, nlg_model=self.nlg_model)
  
    if len(glob.glob(self.checkpoint_prefix + '*')):
      if is_training:
        self.checkpoint.read(self.checkpoint_prefix)
      else:
        self.checkpoint.read(self.checkpoint_prefix).expect_partial()
    else:
      if not is_training:
        raise ValueError('Missing model NPA/NIE')

  def utterance(self, Dt_index):
    batch = []
    for sentence in Dt_index:
      out = []
      for i in sentence:
        #if i == 1: #</s>
        #  break
        out.append(self.common.word_list[i])
      batch.append(out)
      
    return (batch)

  def train(self, mode):

    random_positive_count = int(self.common.batch_size / 2)
    random_negative_count = self.common.batch_size - random_positive_count
  
    epoch = int(self.optimizer.iterations / self.common.batches_per_epoch) + 1

    dst = DST(self.common, FLAGS.slot, False)
    while epoch < self.common.max_epoch+1:
      for _ in range(self.common.batches_per_epoch):
        start_time=time.time()
  
        positive_indices = np.random.choice(len(self.common.utterances_train), self.common.batch_size)

        #input featues, DA is the same regardless target_lot: if to send batch of utterences? 
        bt = dst.predict_batch(self.feature_vectors, self.common.utterances_train, positive_indices)

        tf.print ("DST: ", bt.shape)

        batch_data = generate_npa_examples(self.feature_vectors, self.common.word_vectors, self.common.dialogue_ontology, self.common.utterances_train, positive_indices)

        (batch_xs_full, batch_masks, batch_sys_req, batch_sys_conf_slots, batch_sys_conf_values, batch_delex, batch_target, batch_information_richness, batch_sentenceGroup, batch_offered, batch_requests, batch_goals) = batch_data
  
        with tf.GradientTape() as tape:
          if mode == 'RL':
            per_example_log_policy, Dt_index = self.p_model.rl(batch_delex, batch_masks, batch_target, batch_information_richness, batch_sentenceGroup, bt)
            bleu = np.array([sentence_bleu([list(map(str, list(batch_target[:,1:][i])))], list(map(str, list(Dt_index[:,:-1].numpy()[i])))) for i in range(self.common.batch_size)])

            original_success = [batch_offered[i] and set(batch_requests[i]).issuperset(set(batch_goals[i])) for i in range(self.common.batch_size)]

            generated = self.utterance(Dt_index)

            generated_offered = [False for i in range(self.common.batch_size)]
            for i, sentense in enumerate(generated):
              for r in self.common.restaurant_db:
                if r in sentense:
                  generated_offered[i] = True
                  break

            pure_requestables = ["address", "phone", "postcode"]
            generated_requests = []
            for i, sentense in enumerate(generated):
              requests = []
              for r in pure_requestables:
                if r in sentense:
                  requests.append(r)
              generated_requests.append(requests)
 
            success = []
            for i in range(self.common.batch_size):
              if not batch_offered[i]:
                if generated_offered[i]:
                  success.append(set(generated_requests[i]).issuperset(set(batch_goals[i])))
                else:
                  success.append(False)
              else:
                success.append(set(batch_requests[i] + generated_requests[i]).issuperset(set(batch_goals[i])))

            r_ben = np.array(success).astype(float) - np.array(original_success).astype(float)

            tf.print("Req before turn", batch_requests[0])
            tf.print("Goal requestabl", batch_goals[0])
            tf.print("Restaurant Offe", batch_offered[0])
            tf.print("New requests   ", generated_requests[0])
            tf.print("Success before ", original_success[0])
            tf.print("Success after  ", success[0])
            tf.print("Benefit        ", r_ben[0])

            reward = r_ben + 0.5*bleu-0.1

            total_loss = -tf.reduce_mean(reward * per_example_log_policy)

            gradients = tape.gradient(total_loss, self.nie_model.variables + self.npa_model.variables)

            if (FLAGS.clip_gradients > 0):
              gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.clip_gradients)
  
            self.optimizer.apply_gradients(zip(gradients, self.nie_model.variables + self.npa_model.variables))
  
            tf.print ("T: ", '|'.join(self.utterance(batch_target)[0]))
            tf.print ("G: ", '|'.join(generated[0]))

            tf.print("loss:", self.optimizer.iterations, epoch, total_loss, tf.reduce_mean(reward), tf.reduce_mean(r_ben), self.learning_rate_fn(self.optimizer.iterations), time.time() - start_time, output_stream=sys.stderr, summarize=-1)

          else:

            total_p_loss, total_KL_loss, total_q_loss, total_base_loss, Dt_index, r, log_q_prob, reward = self.p_model(batch_delex, batch_masks, batch_target, batch_information_richness, batch_sentenceGroup, bt)

            generated = self.utterance(Dt_index)

            gradients = tape.gradient(total_p_loss + total_KL_loss + total_q_loss + total_base_loss, self.p_model.variables)

            if (FLAGS.clip_gradients > 0):
              gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.clip_gradients)
  
            self.optimizer.apply_gradients(zip(gradients, self.p_model.variables))
  
            # 0.1 is for KL . another 0.1 is for loss improvement
            lower_bound = -total_p_loss - 0.1*total_KL_loss

            tf.print ("T: ", '|'.join(self.utterance(batch_target)[0]))
            tf.print ("G: ", '|'.join(generated[0]))

            tf.print("loss:", self.optimizer.iterations, epoch, total_p_loss, total_KL_loss, total_p_loss + total_KL_loss, tf.reduce_mean(r), tf.reduce_mean(log_q_prob), total_q_loss, total_base_loss, tf.reduce_mean(reward), lower_bound, self.learning_rate_fn(self.optimizer.iterations), time.time() - start_time, output_stream=sys.stderr, summarize=-1)
  
        if self.optimizer.iterations % FLAGS.save_batches == 0 or self.optimizer.iterations == 1:
            self.checkpoint.write(file_prefix=self.checkpoint_prefix)
  
        epoch = int(self.optimizer.iterations / self.common.batches_per_epoch) + 1
  
def main():  
  common = Common()
  if FLAGS.action == 'TRAIN_DST':
    DST(common, FLAGS.slot, True).train()
  if FLAGS.action == 'TRAIN_VIN':
    #with tf.device('/cpu:0'):
    m = VIN(common)
    m.train('')
  if FLAGS.action == 'TRAIN_RL':
    m = VIN(common)
    m.train('RL')
  elif FLAGS.action == 'TEST_DST':
    with tf.device('/cpu:0'):
      DST(common, FLAGS.slot, False).test()
  elif FLAGS.action == 'PREDICT_DST':
    with tf.device('/cpu:0'):
      DST(common, FLAGS.slot, False).predict()

if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument('--config_file', type=str, default='config/predict.tfrecords',
            help='Predict file location in google storage.')
  parser.add_argument('--sentense_length', type=int, default=40,
            help='Max sentense lenght.')
  parser.add_argument('--hidden_size', type=int, default=50,
            help='Embedding/hidden size.')
  parser.add_argument('--dst_hidden_size', type=int, default=100,
            help='hidden size in DST.')
  parser.add_argument('--delex_threshold', type=float, default=0.9,
            help='Threshold to replace sentence slot/values embeddings with ontology slot/values.')
  parser.add_argument('--num_pairs', type=int, default=112,
            help='Number of inform and request slot/values pairs including dontcare and none.')
  parser.add_argument('--num_informable_pairs', type=int, default=102,
            help='Number of inform slot/values pairs including dontcare and none.')
  parser.add_argument('--num_slots', type=int, default=4,
            help='Number of inform slot/values pairs including dontcare and none.')
  parser.add_argument('--num_actions', type=int, default=70,
            help='These are dialog actions as in dialogs.')
  parser.add_argument('--dkl_lambda', type=float, default=0.1,
            help='Hyperparameter to regularize KL divergence.')
  parser.add_argument('--eta', type=float, default=0.1,
            help='Tradeoff between supervised and unsupervised training.')

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
  parser.add_argument('--learning_rate', type=float, default=3e-3,
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
  parser.add_argument('--action', default='PREDICT', choices=['TRAIN_DST', 'TRAIN_VIN', 'TRAIN_RL', 'TEST_DST', 'PREDICT_DST'],
            help='An action to execure.')
  parser.add_argument('--slot', choices=['request', 'food', 'price range', 'area'],
            help='Slot to train.')

  FLAGS, unparsed = parser.parse_known_args()

  tf.print ("Running with parameters: {}".format(FLAGS))

  main()
