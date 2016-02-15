import sys


codegit_root = '/home/intuinno/codegit'

sys.path.insert(0, codegit_root)

from anandlib.dl.caffe_cnn import *
import pandas as pd
import numpy as np
import os
import scipy
import json
import cPickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TreebankWordTokenizer
import pdb

TRAIN_SIZE = 6000
TEST_SIZE = 1000

annotation_path = '/home/intuinno/project/pointTeach/data/Flicker8k/Flickr8k.token.txt'
vgg_deploy_path = '/home/intuinno/codegit/caffe/models/vgg_ilsvrc_19/VGG_ILSVRC_19_layers_deploy.prototxt'
vgg_model_path  = '/home/intuinno/codegit/caffe/models/vgg_ilsvrc_19/VGG_ILSVRC_19_layers.caffemodel'
flickr_image_path = '/home/intuinno/project/pointTeach/data/Flicker8k/preprocessedImages'
feat_path='feat/flickr8k'

cnn = CNN(deploy=vgg_deploy_path,
          model=vgg_model_path,
          batch_size=20,
          width=224,
          height=224)

def my_tokenizer(s):
    return s.split()
pdb.set_trace()

annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
annotations['image_num'] = annotations['image'].map(lambda x: x.split('#')[1])
annotations['image'] = annotations['image'].map(lambda x: os.path.join(flickr_image_path,x.split('#')[0]))

captions = annotations['caption'].values

vectorizer = CountVectorizer(lowercase=False, analyzer=str.split).fit(captions)
dictionary = vectorizer.vocabulary_
dictionary_series = pd.Series(dictionary.values(), index=dictionary.keys()) + 2

dictionary = dictionary_series.to_dict()

pdb.set_trace()
# Sort dictionary in descending order
from collections import OrderedDict
dictionary = OrderedDict(sorted(dictionary.items(), key=lambda x:x[1], reverse=True))

with open('data/flickr8k/dictionary.pkl', 'wb') as f:
    cPickle.dump(dictionary, f)


