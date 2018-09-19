import numpy as np


def read_glove_word_vecs(filename):
    print('loading {} ...'.format(filename))
    word_vec_dict = dict()
    f = open(filename, encoding='utf-8')
    for line in f:
        parts = line.strip().split(' ')
        vec = [float(v) for v in parts[1:]]
        word_vec_dict[parts[0]] = np.asarray(vec, np.float32)
    f.close()
    print('done')
    return word_vec_dict
