# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 22:11:12 2016

@author: kalifou
"""

import os
import sys
import re

dirname = sys.argv[1]

for file in os.listdir(dirname):
    
    if file.endswith(".txt"):
        fname = os.path.join(dirname, file)        
        
        with open(fname, 'r') as f:
            read_data = f.read()
            tmp = re.sub(r'(.\w{2,})\. ', r'\1.\n',read_data)
        f.closed

        with open(fname, 'w') as f:
            f.write(tmp)
        f.closed
        