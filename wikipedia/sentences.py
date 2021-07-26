#!/usr/bin/env python3

from io import open
import mmap # https://stackoverflow.com/a/11159418

import re

# download wikipedia:
#   https://dumps.wikimedia.org/eswiki/20210720/ (e.g. "eswiki-20210720-pages-articles-multistream1.xml-p1p159400.bz2")
#   https://dumps.wikimedia.org/ptwiki/20210720/
path = './data' # name of file (after being extracted)

with open(path,'r', encoding='UTF-8') as f:
    m = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) # store file as bytes

    # https://pymotw.com/3/mmap/#regular-expressions
    pattern = re.compile(rb'ferrocarril', re.IGNORECASE)
    res = pattern.findall(m)
    print("num results = {}".format(len(res)))
    print("first result = ")
    print(res[0])

    exit(1)

    print('------')
    # mapping = dict((line.strip().split(' ') for line in f if line))
    mapping = [line.strip().split(' ') for line in f if line]
    print(mapping[0])
    exit(1)
