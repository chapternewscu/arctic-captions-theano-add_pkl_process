import sys
sys.path.insert(0, '../../')

from capgen import train

# training
train(dataset='flickr30k')
