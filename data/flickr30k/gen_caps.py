import sys
sys.path.insert(0, '/home/ubuntu/arctic-captions')

from capgen import train

# training
train(dataset='flickr30k')
