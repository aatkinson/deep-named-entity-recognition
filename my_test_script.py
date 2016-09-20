#!/usr/bin/python
import sys

from data_util import DataUtil
from ner_model import NERModel

WORDVEC_FILEPATH = "wordvecs.txt"
TAGGED_NEWS_FILEPATH = "news_tagged_data.txt"
SAVED_MODEL_FILEPATH = "model_blstm_150_150_ep50.h5"
NEWS_DATA_FILEPATH = "news_tagged_data.txt"
EXTRA_LOGGING = False
PRINT_BAD = True

if __name__ == "__main__":

    if len(sys.argv) > 1:
        n_samples = int(sys.argv[1])
    else:
        n_samples = sys.maxint

    reader = DataUtil(WORDVEC_FILEPATH, TAGGED_NEWS_FILEPATH)
    nermodel = NERModel(reader)

    nermodel.load(SAVED_MODEL_FILEPATH)

    with open(NEWS_DATA_FILEPATH, 'r') as f:
        cur_sentence = []
        cur_tags = []

        samples_read = 0
        total_frames = 0
        total_matched_frames = 0
        total_correct_preditions = 0

        for line in f:
            line = line.strip()
            
            if line == "":

                cur_prediction = nermodel.predict_sentence(cur_sentence)

                if EXTRA_LOGGING:
                    print "Words: {0}".format(cur_sentence)
                    print "Tags : {0}".format(cur_tags)
                    print "Preds: {0}\n".format(cur_prediction)

                if not(len(cur_prediction) == len(cur_sentence) == len(cur_tags)):
                    print "MISMATCH"
                    print cur_sentence
                    print cur_tags
                    print cur_prediction
                    raise Exception("Mismatched pred/tag lengths!")

                matched_frames = 0
                for ix in xrange(len(cur_prediction)):
                    if cur_prediction[ix] == cur_tags[ix]:
                        matched_frames += 1
                total_frames += len(cur_prediction)
                total_matched_frames += matched_frames

                if len(cur_prediction) == matched_frames:
                    total_correct_preditions += 1
                elif PRINT_BAD:
                    print "\nBad Prediction!"
                    print "Words: {0}".format(cur_sentence)
                    print "Tags : {0}".format(cur_tags)
                    print "Preds: {0}\n".format(cur_prediction)

                cur_sentence = []
                cur_tags = []
                samples_read += 1

                if samples_read % 100 == 0:
                    print "Samples read = {0}".format(samples_read)

                if samples_read >= n_samples:
                    break
            
            else:
                word, tag = line.split('\t')
                cur_sentence.append(word)
                cur_tags.append(tag)

    print "~~~ Summary ~~~"
    print "# samples read = {0}".format(samples_read)
    print "Correctly classified samples = {0:.4f}".format(float(total_correct_preditions)/samples_read)
    print "Correctly classified frames = {0:.4f}".format(float(total_matched_frames)/total_frames)
