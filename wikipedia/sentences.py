#!/usr/bin/env python3

from io import open
import mmap # https://stackoverflow.com/a/11159418

import re

# download wikipedia:
#   https://dumps.wikimedia.org/eswiki/20210720/ (e.g. "eswiki-20210720-pages-articles-multistream1.xml-p1p159400.bz2")
#   https://dumps.wikimedia.org/ptwiki/20210720/
path = './pt_plaintext.txt' # name of file (after being extracted)

def getExample(m, sIndex, eIndex):
    """
    return a string with additional context on either side of the given search result
    """
    sIndex = max(0, sIndex - 500)
    eIndex = min(len(m) - 1, eIndex + 250)
    sample = m[sIndex:eIndex].decode().replace('\n', '')

    sIndex = sample.find('. ') # start the sample at the beginning of a sentence
    try:
        eIndex = sample.rindex('. ') # end the sample at the end of a sentence
    except ValueError as e:
        return None
    if sIndex == -1 or eIndex == -1:
        return None
    return sample[(sIndex + 2):(eIndex + 1)]

target = ' têm '.encode() # best to put spaces around it (to avoid submatches)

with open(path,'r', encoding='UTF-8') as f:
    m = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) # store file as bytes

    # https://pymotw.com/3/mmap/#regular-expressions
    pattern = re.compile(target, re.IGNORECASE)

    count = 0
    r = pattern.search(m)
    while r and count < 10:
        text = m[r.start():r.end()].decode() 
        sample = getExample(m, r.start(), r.end() - 1)
        if sample:
            count += 1
            print("\n\n({0}, {1}) ~ {2}".format(r.start(), r.end() - 1, text))
            print(sample)
        r = pattern.search(m, r.start() + 1)

    #res = pattern.findall(m)
    #print("num results = {}".format(len(res)))

    exit(1)
