# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 22:11:12 2016

@author: kalifou
"""

import os
import sys
import re
import codecs

dirname = sys.argv[1]

for file in os.listdir(dirname):
    
    if file.endswith(".txt"):
        fname = os.path.join(dirname, file)        
        
        with codecs.open(fname, 'r','utf-8') as f:
            read_data = f.read()

        f.closed

        with open(fname, 'w') as f:
            inter = read_data.encode("utf-8")

            f.write(inter)
            print 'converted'
        f.closed
