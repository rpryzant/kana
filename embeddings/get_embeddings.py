"""
grabs pretrained vecs for items in its vocab

python get_embeddings.py /Volumes/datasets/nmt/ASPEC/tokenized_kytea/vocabs/train1.8000.ja.txt /Volumes/datasets/nmt/ASPEC/tokenized_kytea/vocabs/train1.8000.en.txt /Users/rpryzant/kana/data/polyglot_vecs/polyglot-ja.pkl /Users/rpryzant/kana/data/polyglot_vecs/polyglot-en.pkl /Users/rpryzant/kana/data/kanjidic2.xml

"""
import sys
import pickle
import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict
import random 
np.random.seed(1)

reload(sys)  
sys.setdefaultencoding('utf8')

ja_vocab = sys.argv[1]
en_vocab = sys.argv[2]
ja_vecs_pkl = sys.argv[3]
en_vecs_pkl = sys.argv[4]
kanjidic = sys.argv[5]

def flatten(l):
    return [x for li in l for x in li]

# setup {kanji => [radical meanings]}
kanjidic_root = ET.parse(kanjidic).getroot()

kanji_all = defaultdict(list)
kanji_singular = defaultdict(list)
for child in kanjidic_root.iter('character'):
    for x in child.iter('meaning'):
        if x.attrib == {}:
            rad = child.find('literal').text
            for mi in x.text.split():
                kanji_all[rad].append(mi)
            if len(x.text.split()) < 2:
                kanji_singular[rad].append(x.text)
kanji_first = {k: [v[0]] for k, v in kanji_singular.iteritems()} 
kanji_random = {k: [random.choice(v)] for k, v in kanji_singular.iteritems()}

# setup {vocab => word2vec embedding}
toks, embeddings = pickle.load(open(ja_vecs_pkl))
ja_word2vec = {
    tok.lower(): embedding \
    for tok, embedding in zip(toks, embeddings)
}
# setup {vocab => word2vec embedding}
toks, embeddings = pickle.load(open(en_vecs_pkl))
en_word2vec = {
    tok.lower(): embedding \
    for tok, embedding in zip(toks, embeddings)
}
# get embedding len
n = len(en_word2vec['the'])


def random_embedding(v):
    return list(np.random.uniform(low=-0.001, high=0.001, size=n))


def word2vec_embedding(v, word2vecs):
    if v in word2vecs:
        return list(word2vecs[v])
    else:
        return list(np.random.uniform(low=-0.001, high=0.001, size=n))


def kanji_embedding(v, word2vecs, kanji_dict):
    out = []
    for rad, meanings in kanji_dict.iteritems():
        if rad in v:
            for m in meanings:
                if m in word2vecs:
                    out.append(word2vecs[m])
    if len(out) == 0:
        out.append(np.random.uniform(low=-0.001, high=0.001, size=n))
    return list(np.mean(out, axis=0))


def hybrid_embedding(v, word2vecs, kanji_dict):
    if v in word2vecs:
        return list(word2vecs[v])
    else:
        return kanji_embedding(v, word2vecs, kanji_dict)




for v in open(ja_vocab):
    v = v.strip()
    # TODO -- make and write them all
    
