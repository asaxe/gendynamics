import argparse
from subprocess import Popen, PIPE
import os

parser = argparse.ArgumentParser()

parser.add_argument('-paramfile', type=argparse.FileType('r'))
parser.add_argument('-line', type=int)


settings = parser.parse_args(); 

for l, line in enumerate(settings.paramfile):
    if l == settings.line:
        #print line
        #call(line, shell=True)
        p = Popen(line, shell=True, stdout=PIPE, stderr=PIPE)
        p.communicate()
        



