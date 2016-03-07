import argparse
import numpy
import cPickle as pkl
import pdb
from capgen import init_params, get_dataset, build_sampler, gen_sample
from util import load_params, init_tparams

# single instance of a sampling process
def gen_model(idx, context, model, options, k, normalize, word_idict, sampling):
    import theano
    from theano import tensor
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

    trng = RandomStreams(1234)
    # this is zero indicate we are not using dropout in the graph
    use_noise = theano.shared(numpy.float32(0.), name='use_noise')

    # get the parameters
    params = init_params(options)
    params = load_params(model, params)
    tparams = init_tparams(params)

    # build the sampling computational graph
    # see capgen.py for more detailed explanations
    f_init, f_next = build_sampler(tparams, options, use_noise, trng, sampling=sampling)

    def _gencap(cc0):
        sample, score = gen_sample(tparams, f_init, f_next, cc0, options,
                                   trng=trng, k=k, maxlen=200, stochastic=False)
        # adjust for length bias
        if normalize:
            lengths = numpy.array([len(s) for s in sample])
            score = score / lengths
        sidx = numpy.argmin(score)
        return sample[sidx]

    seq = _gencap(context)

    return (idx, seq)


def main(model, saveto, k=5, normalize=False, zero_pad=False, datasets='dev,test', sampling=False, pkl_name=None):
# load model model_options
    if pkl_name is None:
        pkl_name = model
    with open('%s.pkl'% pkl_name, 'rb') as f:
        options = pkl.load(f)

    # fetch data, skip ones we aren't using to save time
    load_data, prepare_data = get_dataset(options['dataset'])
    train, valid, test, worddict = load_data(path='./data/coco/', load_train=True if 'train' in datasets else False,
                                             load_dev=True if 'dev' in datasets else False,
                                             load_test=True if 'test' in datasets else False)
    
    # import pdb; pdb.set_trace()
    # <eos> means end of sequence (aka periods), UNK means unknown
    word_idict = dict()
    for kk, vv in worddict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'


    # index -> words
    def _seqs2words(caps):
        capsw = []
        for cc in caps:
            ww = []
            for w in cc:
                if w == 0:
                    break
                ww.append(word_idict[w])
            capsw.append(' '.join(ww))
        return capsw

    # process all dev examples
    def _process_examples(contexts):
        caps = [None] * contexts.shape[0]
        for idx, ctx in enumerate(contexts):
            cc = ctx.todense().reshape([14*14,512])
            if zero_pad:
                cc0 = numpy.zeros((cc.shape[0]+1, cc.shape[1])).astype('float32')
                cc0[:-1,:] = cc
            else:
                cc0 = cc
            resp = gen_model(idx, cc0, model, options, k, normalize, word_idict, sampling)
            caps[resp[0]] = resp[1]
            print 'Sample ', (idx+1), '/', contexts.shape[0], ' Done'
            print resp[1]
        return caps

    ds = datasets.strip().split(',')

    # send all the features for the various datasets
    for dd in ds:
        if dd == 'train':
            print 'Training Set...',
            caps = _seqs2words(_process_examples(train[1]))
            # import pdb; pdb.set_trace()
            with open(saveto+'.train.txt', 'w') as f:
                print >>f, '\n'.join(caps)
            print 'Done'
        if dd == 'dev':
            print 'Development Set...',
            caps = _seqs2words(_process_examples(valid[1]))
            # import pdb; pdb.set_trace()
            with open(saveto+'.dev.txt', 'w') as f:
                print >>f, '\n'.join(caps)
            print 'Done'
        if dd == 'test':
            print 'Test Set...',
            caps = _seqs2words(_process_examples(test[1]))
            with open(saveto+'.test.txt', 'w') as f:
                print >>f, '\n'.join(caps)
            print 'Done'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=1)
    parser.add_argument('-sampling', action="store_true", default=False) # this only matters for hard attention
    parser.add_argument('-n', action="store_true", default=False)
    parser.add_argument('-z', action="store_true", default=False)
    parser.add_argument('-d', type=str, default='dev,test')
    parser.add_argument('-pkl_name', type=str, default=None, help="name of pickle file (without the .pkl)")
    parser.add_argument('model', type=str)
    parser.add_argument('saveto', type=str)

    args = parser.parse_args()
    main(args.model, args.saveto, k=args.k, zero_pad=args.z, pkl_name=args.pkl_name, normalize=args.n, datasets=args.d, sampling=args.sampling)
