#!/usr/bin/python
from data_util import DataUtil

WORDVEC_FILEPATH = "wordvecs.txt"
TAGGED_NEWS_FILEPATH = "news_tagged_data.txt"

if __name__ == "__main__":

    reader = DataUtil(WORDVEC_FILEPATH, TAGGED_NEWS_FILEPATH)
    