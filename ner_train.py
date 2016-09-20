#!/usr/bin/python
import sys
from data_util import DataUtil
from ner_model import NERModel

WORDVEC_FILEPATH = "wordvecs.txt"
TAGGED_NEWS_FILEPATH = "news_tagged_data.txt"

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print "ERROR: need model destination filepath!"
        sys.exit(1)

    if len(sys.argv) > 2:
        layer_arg = int(sys.argv[2])
    else:
        layer_arg = 2

    if len(sys.argv) > 3:
        ep_arg = int(sys.argv[3])
    else:
        ep_arg = 20

    # Read the data
    print ">> Initializing data..."
    reader = DataUtil(WORDVEC_FILEPATH, TAGGED_NEWS_FILEPATH)
    X,Y = reader.get_data()
    print X.shape
    print Y.shape

    # Train the model
    print ">> Training model... epochs = {0}, layers = {1}".format(ep_arg,layer_arg)
    nermodel = NERModel(reader)
    nermodel.train(epochs=ep_arg, layers=layer_arg)

    # Evaluate the model
    print ">> Evaluating model..."
    nermodel.evaluate()

    # Save the model
    print ">> Saving model..."
    nermodel.save(sys.argv[1])

    print ">> Done."