#!/usr/bin/python
import sys

from data_util import DataUtil
from ner_model import NERModel

WORDVEC_FILEPATH = "wordvecs.txt"
TAGGED_NEWS_FILEPATH = "news_tagged_data.txt"
SAVED_MODEL_FILEPATH = "model_blstm_150_150_ep50.h5"

if __name__ == "__main__":

    reader = DataUtil(WORDVEC_FILEPATH, TAGGED_NEWS_FILEPATH)
    nermodel = NERModel(reader)

    nermodel.load(SAVED_MODEL_FILEPATH)

    # Model takes ~35s to load on my machine. stdin is still buffered when loading,
    # and put into model after, but the query (e.g. "your sentence")
    # doesn't proceed the results. 
    # This may mess up the output format your expect.
    #
    # Concretely, you may end up with something like this:
    #
    # this is my query <-- input from stdin or from file direction 
    #
    # <... some time passes while model is still loading>
    #
    # Type a query (type "exit" to exit):
    #
    # this  TAG
    # is    TAG
    # my    TAG
    # query TAG

    while True:
        print "Type a query (type \"exit\" to exit):"
        try:
            raw_s = raw_input().strip()
        except EOFError as e:
            sys.exit(0)

        if raw_s == "exit":
            break

        if not raw_s:
            continue

        print ""
        sentence = raw_s.split()
        
        # Max inference time I've seen on my machine is 3s, usually near-instant
        prediction = nermodel.predict_sentence(sentence)
        for ix in xrange(len(sentence)):
            print "{0}\t{1}".format(sentence[ix], prediction[ix])
        print ""
