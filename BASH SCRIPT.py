import os
import sys

path = os.getcwd() + "/" + "data/123-4Experiments"
pkls = os.listdir(path)
evalsfile = open("Evals.py", "r")
evals = evalsfile.read()
for pkl in pkls:
    sys.argv = ["filename", path + "/" + pkl]
    exec(evals)
evalsfile.close()