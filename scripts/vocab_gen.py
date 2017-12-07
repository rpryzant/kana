"""
generates vocab

"""

from collections import Counter
import sys

c  = Counter(open(sys.argv[1]).read().split())

for x, _ in  c.most_common(32000):
    print x
