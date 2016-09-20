#!/usr/bin/python
################################################################################
# Data processing for NER challenge
# https://www.dropbox.com/sh/40nkmagam5513yt/AACKk5DMpIF9qurj-dQIJcMga
# 1624 / 11100 words are unknown (15%)
################################################################################
from keras.preprocessing import sequence
import numpy as np
import pandas as pd

class DataUtil:

    def __init__ (self, wordvec_filepath=None, news_filepath=None):
        # Some constants
        self.DEFAULT_N_CLASSES = 10
        self.DEFAULT_N_FEATURES = 300
        self.DEFAULT_MAX_SEQ_LENGTH = 29
        # Other stuff
        self.wordvecs = None
        self.word_to_ix_map = {}
        self.n_features = 0
        self.n_tag_classes = 0
        self.n_sentences_all = 0
        self.tag_vector_map = {}
        self.max_sentence_len = 0
        self.all_X = []
        self.all_Y = []
        if wordvec_filepath and news_filepath:
            self.read_and_parse_data(wordvec_filepath, news_filepath)

    def read_and_parse_data (self, wordvec_filepath, news_filepath, skip_unknown_words=False):

        # Read word vectors file and create map to match words to vectors
        self.wordvecs = pd.read_table(wordvec_filepath, sep='\t', header=None)
        self.word_to_ix_map = {}
        for ix, row in self.wordvecs.iterrows():
            self.word_to_ix_map[row[0]] = ix
        self.wordvecs = self.wordvecs.drop(self.wordvecs.columns[[0,-1]], axis=1).as_matrix()
        #print self._wordvecs.shape
        self.n_features = len(self.wordvecs[0])
        #print self._n_features

        # Read in training data and create map to match tag classes to tags
        # Create tag to class index map first because we need total # classes before we can
        with open(news_filepath, 'r') as f:
            self.n_tag_classes = self.DEFAULT_N_CLASSES
            self.tag_vector_map = {}
            tag_class_id = 0
            raw_news_data = []
            raw_news_words = []
            raw_news_tags = []        

            # Process all lines in the file
            for line in f:
                line = line.strip()
                if not line:
                    raw_news_data.append( (tuple(raw_news_words), tuple(raw_news_tags)) )
                    raw_news_words = []
                    raw_news_tags = []
                    continue
                word, tag = line.split('\t')
                raw_news_words.append(word)
                raw_news_tags.append(tag)
                if tag not in self.tag_vector_map:
                    one_hot_vec = np.zeros(self.DEFAULT_N_CLASSES, dtype=np.int32)
                    one_hot_vec[tag_class_id] = 1
                    self.tag_vector_map[tag] = tuple(one_hot_vec)
                    self.tag_vector_map[tuple(one_hot_vec)] = tag
                    tag_class_id += 1

        # Add nil class
        one_hot_vec = np.zeros(self.DEFAULT_N_CLASSES, dtype=np.int32)
        one_hot_vec[tag_class_id] = 1
        self.tag_vector_map['NIL'] = tuple(one_hot_vec)
        self.tag_vector_map[tuple(one_hot_vec)] = 'NIL'

        self.n_sentences_all = len(raw_news_data)

        # Build the data as required for training
        self.max_sentence_len = 0
        for seq in raw_news_data:
            if len(seq[0]) > self.max_sentence_len:
                self.max_sentence_len = len(seq[0])

        self.all_X, self.all_Y = [], []
        unk_words = []
        for words, tags in raw_news_data:
            elem_wordvecs, elem_tags = [], []
            
            for ix in xrange(len(words)):
                w = words[ix]
                t = tags[ix]
                if w in self.word_to_ix_map:
                    elem_wordvecs.append(self.wordvecs[self.word_to_ix_map[w]])
                    elem_tags.append(list(self.tag_vector_map[t]))

                # Ignore unknown words, removing from dataset
                elif skip_unknown_words:
                    unk_words.append(w)
                    continue
                
                # Randomly select a 300-elem vector for unknown words
                else:
                    unk_words.append(w)
                    new_wv = 2*np.random.randn(300)-1 # sample from normal distn
                    norm_const = np.linalg.norm(new_wv)
                    new_wv /= norm_const
                    self.word_to_ix_map[w] = self.wordvecs.shape[0]
                    self.wordvecs = np.vstack((self.wordvecs, new_wv))
                    elem_wordvecs.append(new_wv)
                    elem_tags.append(list(self.tag_vector_map[t]))

            # Pad the sequences for missing entries to make them all the same length
            nil_X = np.zeros(300)
            nil_Y = np.array(self.tag_vector_map['NIL'])
            pad_length = self.max_sentence_len - len(elem_wordvecs)
            self.all_X.append( ((pad_length)*[nil_X]) + elem_wordvecs)
            self.all_Y.append( ((pad_length)*[nil_Y]) + elem_tags)

        self.all_X = np.array(self.all_X)
        self.all_Y = np.array(self.all_Y)
        
        #print "UNKNOWN WORDS " + str(unk_words)
        #print "UNK WORD COUNT " + str(len(unk_words))
        #print "TOTAL WORDS " + str(self.wordvecs.shape[0])

        return (self.all_X, self.all_Y)

    def get_data (self):
        return (self.all_X, self.all_Y)

    def encode_sentence (self, sentence, skip_unknown_words=False):
        X = []
        outer_X = []
        for word in sentence:
            if word in self.word_to_ix_map:
                X.append(self.wordvecs[ self.word_to_ix_map[word] ])

            elif skip_unknown_words:
                continue
                
            else:
                new_wv = 2*np.random.randn(300)-1 # sample from normal distn
                norm_const = np.linalg.norm(new_wv)
                new_wv /= norm_const
                X.append(new_wv)

        outer_X.append(X)
        return sequence.pad_sequences(outer_X, maxlen=self.max_sentence_len, dtype=np.float64)

    def decode_prediction_sequence (self, pred_seq):
        pred_tags = []
        for class_prs in pred_seq:
            class_vec = np.zeros(10, dtype=np.int32)
            class_vec[ np.argmax(class_prs) ] = 1
            if tuple(class_vec.tolist()) in self.tag_vector_map:
                pred_tags.append(self.tag_vector_map[tuple(class_vec.tolist())])
        return pred_tags
