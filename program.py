# import os
from glob import glob
import cv2
import sys
import re
import numpy as np
from sklearn import svm
from skimage.feature import hog
import matplotlib.pylab as plt
from tkinter import *
# import matplotlib.ticker as ticker

# global variables
DIR = "/home/cesar/Desktop/database/" # linux
# DIR = "D:/543f75gw/SSIG-SegPlate/" # windows

# A BASE DE DADOS CONTEM IMAGENS DE CARROS DE FREQUENTANTES DA UFMG #
#     OS AUTORES PEDIRAM PARA NÃO COMPARTILHAR A BASE DE DADOS      #

letters = "abcdefghijklmnopqrstuvwxyz"
digits = "0123456789"

bin_n = 8 # Number of bins

svms = {}
samples = {}

def main(argv):
    init() # initialize all svms
    train() # traning
    test() # testing

def init():
    for c in letters + digits:
        svms[c] = svm.SVC(probability=True, kernel='rbf', C=0.5, gamma=0.5)

def train():
    tracks = glob(DIR + "training/*/")
    responses = {}
    positives = {}

    N = 0
    for c in letters + digits:
        samples[c] = []
        responses[c] = []
        positives[c] = 0

    print("get data for training...")
	# the training/test database is divided by tracks
	# each track have a collection of frames (images)
    for track in tracks:
        files = glob(track + "/*.png")
        for f in files: # each image of the track
            img = cv2.imread(f, 0) # read image
            notes = parseNotations(f.replace(".png", ".txt")) # get image notes
            text = notes["text"].replace("-","").lower() # characters of the plate
            for i in range(0,len(text)):
                N += 1
                col = letters if (i < 3) else digits
                rect = notes["position_chars"][i]
                for c in col:
                    response = 1 if (c == text[i]) else -1 # one against all
                    if (c == text[i]) positives[c] += 1
                    nimg = cv2.resize(img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]], (24, 32)) # normalize images to hog describer
                    hist = hog(nimg, block_norm='L2-Hys') # hog describer
                    samples[c].append(hist)
                    responses[c].append(response)
					
    print("start training...")
    for c in letters + digits:
        trainData = np.float32(samples[c]) # Convert objects to Numpy Objects
        labels = np.array(responses[c])
        svms[c].fit(trainData, labels) # train svm

def test():
    tracks = glob(DIR + "training/*/")
    responses = {}

    n_tests = 0
    n_errors = 0
    confusion = [[0 for x in range(len(letters + digits))] for y in range(len(letters + digits))]
    conf_index = {}
    for i in range(len(letters + digits)):
        conf_index[(letters + digits)[i]] = i

    print("get data for testing...")
    for track in tracks:
        files = glob(track + "/*.png")
        for f in files: # each image of the track
            img = cv2.imread(f, 0) # read the image
            notes = parseNotations(f.replace(".png", ".txt")) # get image notes
            text = notes["text"].replace("-","").lower() # plate characters
            for i in range(0,len(text)):
                col = letters if (i < 3) else digits
                rect = notes["position_chars"][i]
                results = []
                for c in col:
                    nimg = cv2.resize(img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]], (24, 32)) # normalize images
                    hist = hog(nimg, block_norm='L2-Hys')
                    testData = np.float32([hist]) # test
                    results.append((c, svms[c].predict_proba(testData)[0][1]))
                predicted = oneAgainstAll(results) # get final answer using one-against-all
                confusion[conf_index[text[i]]][conf_index[predicted]] += 1 # feed confusion matrix
                n_tests += 1
                n_errors += 0 if (predicted == text[i]) else 1 # compare real vs. predicted
            
    print("{} tests. {} mistakes.".format(n_tests, n_errors))
    m = np.array(confusion, np.int32)

	# plot confusion matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(m, interpolation='nearest')
    fig.colorbar(cax)
    ax.set_xticklabels(list(letters + digits))
    ax.set_yticklabels(list(letters + digits))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show(block=True)

def oneAgainstAll(results):
    better = 0
    for i in range(0, len(results)):
        if results[i][1] > results[better][1]:
            better = i
    return results[better][0]

def parseNotations(location): # parse notes file
    chars = []
    text_pattern = re.compile("text")
    plate_pattern = re.compile("position_plate")
    chars_pattern = re.compile("char[0-9]")
    with open(location, "rb") as f:
        for line in f:
            line = line.decode("utf-8").strip()
            if text_pattern.match(line):
                text = line.replace("text:","").strip()
            elif plate_pattern.match(line):
                numbers = line.replace("position_plate:", "").strip().split(" ")
                plate_position = tuple(map(lambda x: int(x), numbers))
            elif chars_pattern.match(line):
                numbers = line[7:].strip().split(" ")
                chars.append(tuple(map(lambda x: int(x), numbers)))
        f.close()
        return {"text": text, "position_plate": plate_position, "position_chars": chars}

main(sys.argv[1:])