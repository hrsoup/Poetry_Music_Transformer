import collections
import random
import numpy as np
from config import *

class POEMS:
    "poem class"
    def __init__(self, filename, isEvaluate=False):
        """pretreatment"""
        poems = []
        file = open(filename, "r",encoding="utf-8")
        for line in file:  #every line is a poem
            poem = line.strip() #get poem
            poem = poem.replace(' ','')
            if len(poem) < 10 or len(poem) > 512:  #filter poem
                continue
            if '_' in poem or '《' in poem or '[' in poem or '(' in poem or '（' in poem:
                continue
            poem = '[' + poem + ']' #add start and end signs
            poems.append(poem)

        #counting words
        wordFreq = collections.Counter()
        for poem in poems:
            wordFreq.update(poem)

        wordFreq[" "] = -1
        wordPairs = sorted(wordFreq.items(), key = lambda x: -x[1])
        self.words, freq = zip(*wordPairs)
        self.wordNum = len(self.words)

        self.wordToID = dict(zip(self.words, range(self.wordNum))) #word to ID
        poemsVector = [([self.wordToID[word] for word in poem]) for poem in poems] # poem to vector
        self.trainVector = poemsVector
        self.testVector = []


    def generateBatch(self, isTrain=True):
        #padding length to MaxLength
        poemsVector = self.trainVector

        index = np.random.randint(0, len(poemsVector))
        one_poem = poemsVector[index]
        temp = np.full(maxLength, self.wordToID[" "], np.int32) # padding space
        temp[:len(one_poem)] = one_poem
        temp2 = np.copy(temp) 
        temp2[:-1] = temp[1:]
        return temp, temp2
