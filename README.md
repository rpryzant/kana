# kana
This was a little project which I ended up dropping, but I had some results suggesting the idea worked.

People sometimes initialize vocabs using word vectors in small-to-medium-data Japanese-English NMT. We want to show that initializing Kanji radicals with the english word vector for their reading can beat equivalent vectors learned from Japanese data.

data/runs are on jagupard10 & 11

Results are

bye random 17.9
bye w2v 18.3
bye kanji 18.3
bye hybrid 19.0

kytea random 19.4
kytea w2v  19.9
kytea kanji 20.1
kytea hybrid 22.0


