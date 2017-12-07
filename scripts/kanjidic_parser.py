"""
=== DESCRIPTION
parses kanjidic2 and maps kana to a list of english meanings


=== USAGE
python kanjidic_parser.py [kanjidic2 xml]
python kanjidic_parser.py ../data/kanjidic2.xml
"""
from bs4 import BeautifulSoup

import sys

input = open(sys.argv[1]).read()

soup = BeautifulSoup(input)

print soup
