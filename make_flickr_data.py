from anandlib.dl.caffe_cnn import *
import pandas as pd
import numpy as np
import os
import scipy
import json
import cPickle
from sklearn.feature_extraction.text import CountVectorizer
import pdb

TRAIN_SIZE = 25000
TEST_SIZE = 5000

annotation_path = 'data/flickr30k/results_20130124.token'
vgg_deploy_path = 'VGG_ILSVRC_16_layers_deploy.prototxt'
vgg_model_path  = '/home/ubuntu/captionly/cnn_params/VGG_ILSVRC_16_layers.caffemodel'
flickr_image_path = 'data/flickr30k/flickr30k-images'
feat_path='feat/flickr30k'
cnn = CNN(deploy=vgg_deploy_path,
          model=vgg_model_path,
          batch_size=20,
          width=224,
          height=224)

annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
annotations['image_num'] = annotations['image'].map(lambda x: x.split('#')[1])
annotations['image'] = annotations['image'].map(lambda x: os.path.join(flickr_image_path,x.split('#')[0]))

pdb.set_trace()

captions = annotations['caption'].values

vectorizer = CountVectorizer().fit(captions)
dictionary = vectorizer.vocabulary_
dictionary_series = pd.Series(dictionary.values(), index=dictionary.keys()) + 2
dictionary = dictionary_series.to_dict()

with open('data/flickr30k/dictionary.pkl', 'wb') as f:
    cPickle.dump(dictionary, f)

images = pd.Series(annotations['image'].unique())
image_id_dict = pd.Series(np.array(images.index), index=images)

DEV_SIZE = len(images) - TRAIN_SIZE - TEST_SIZE

caption_image_id = annotations['image'].map(lambda x: image_id_dict[x]).values
cap = zip(captions, caption_image_id)

# split up into train, test, and dev
all_idx = range(len(images))
np.random.shuffle(all_idx)
train_idx = all_idx[0:TRAIN_SIZE]
test_idx = all_idx[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]
dev_idx = all_idx[TRAIN_SIZE+TEST_SIZE:]

# training set
images_train = images[train_idx]
image_id_dict_train = image_id_dict[train_idx]
orig_caption_image_id_train = caption_image_id[train_idx]
# Reindex the training set
caption_image_id_train = xrange(TRAIN_SIZE)
captions_train = captions[train_idx]
cap_train = zip(captions_train, caption_image_id_train)

for start, end in zip(range(0, len(images_train)+100, 100), range(100, len(images_train)+100, 100)):
    image_files = images_train[start:end]
    feat = cnn.get_features(image_list=image_files, layers='conv5_3', layer_sizes=[512,14,14])
    if start == 0:
        feat_flatten_list_train = scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))
    else:
        feat_flatten_list_train = scipy.sparse.vstack([feat_flatten_list_train, scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))])

    print "processing images %d to %d" % (start, end)

with open('data/flickr30k/flicker_30k_align.train.pkl', 'wb') as f:
    cPickle.dump(cap_train, f)
    cPickle.dump(feat_flatten_list_train, f)

# test set
images_test = images[test_idx]
image_id_dict_test = image_id_dict[test_idx]
orig_caption_image_id_test = caption_image_id[test_idx]
# Reindex the test set
caption_image_id_test = xrange(TEST_SIZE)
captions_test = captions[test_idx]
cap_test = zip(captions_test, caption_image_id_test)

for start, end in zip(range(0, len(images_test)+100, 100), range(100, len(images_test)+100, 100)):
    image_files = images_test[start:end]
    feat = cnn.get_features(image_list=image_files, layers='conv5_3', layer_sizes=[512,14,14])
    if start == 0:
        feat_flatten_list_test = scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))
    else:
        feat_flatten_list_test = scipy.sparse.vstack([feat_flatten_list_test, scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))])

    print "processing images %d to %d" % (start, end)

with open('data/flickr30k/flicker_30k_align.test.pkl', 'wb') as f:
    cPickle.dump(cap_test, f)
    cPickle.dump(feat_flatten_list_test, f)

# dev set
images_dev = images[dev_idx]
image_id_dict_dev = image_id_dict[dev_idx]
orig_caption_image_id_dev = caption_image_id[dev_idx]
# Reindex the dev set
caption_image_id_dev = xrange(DEV_SIZE)
captions_dev = captions[dev_idx]
cap_dev = zip(captions_dev, caption_image_id_dev)

for start, end in zip(range(0, len(images_dev)+100, 100), range(100, len(images_dev)+100, 100)):
    image_files = images_dev[start:end]
    feat = cnn.get_features(image_list=image_files, layers='conv5_3', layer_sizes=[512,14,14])
    if start == 0:
        feat_flatten_list_dev = scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))
    else:
        feat_flatten_list_dev = scipy.sparse.vstack([feat_flatten_list_dev, scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))])

    print "processing images %d to %d" % (start, end)

with open('data/flickr30k/flicker_30k_align.dev.pkl', 'wb') as f:
    cPickle.dump(cap_dev, f)
    cPickle.dump(feat_flatten_list_dev, f)
