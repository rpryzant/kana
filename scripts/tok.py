"""
=== DESCRIPTION
tokenizes an en rowfile

=== USAGE


for ja
kytea < /Volumes/datasets/nmt/ASPEC/raw/train/train-3_ja.txt > /Volumes/datasets/nmt/ASPEC/tokenized_kytea/train/train-3_ja.txt -out tok
"""
import sys
from nltk.tokenize import word_tokenize

def is_ascii(c):
    return ord(c) < 128

for l in open(sys.argv[1]):
    l = ''.join(c for c in l if is_ascii(c))
    print ' '.join(w.lower() for w in word_tokenize(l.strip()))



