import collections
import random
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from config import *

data_train_dir = '../Music_dataset_after_preprocess/train'
data_train_files = []
data_train_files += [join(data_train_dir, f) for f in listdir(data_train_dir) if isfile(join(data_train_dir, f)) if '.npz' in f]
data_train_files.sort()

data_test_dir = '../Music_dataset_after_preprocess/test'
data_test_files = []
data_test_files += [join(data_test_dir, f) for f in listdir(data_test_dir) if isfile(join(data_test_dir, f)) if '.npz' in f]
data_test_files.sort()

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
        if maxLength >= len(one_poem):
            temp[:len(one_poem)] = one_poem 
        else:
            temp[:len(one_poem)] = one_poem[:maxLength]
        temp2 = np.copy(temp) 
        temp2[:-1] = temp[1:]
        # max_value = self.wordToID[" "] / 2
        # temp = np.array([item / max_value for item in temp])
        # temp2 = np.array([item / max_value for item in temp2])
        return temp, temp2


def get_data(length, data_files, typ, iteration, type):
    if type == 'music':
        if typ == 'train':
            index = np.random.randint(0, len(data_files))
        elif typ == 'test':
            index = iteration
            
        data = np.load(data_files[index])['eventlist']  
        
        # time augmentation
        data[:, 0] *= np.random.uniform(0.80, 1.20)
        
        # absolute time to relative interval
        data[1:, 0] = data[1:, 0] - data[:-1, 0]
        data[0, 0] = 0
        
        # discretize interval into IntervalDim
        data[:, 0] = np.clip(np.round(data[:, 0] * IntervalDim), 0, IntervalDim - 1)
        
        # Note augmentation
        data[:, 2] += np.random.randint(-6, 6)
        data[:, 2] = np.clip(data[:, 2], 0, NoteOnDim - 1)
        
        eventlist = []
        for d in data:
            # append interval
            interval = d[0]
            eventlist.append(interval)
        
            # note on case
            if d[1] == 1:
                velocity = (d[3] / 128) * VelocityDim + VelocityOffset
                note = d[2] + NoteOnOffset
                eventlist.append(velocity)
                eventlist.append(note)
                
            # note off case
            elif d[1] == 0:
                note = d[2] + NoteOffOffset
                eventlist.append(note)
            # CC
            elif d[1] == 2:
                event = CCOffset + d[3]
                eventlist.append(event)
                
        eventlist = np.array(eventlist).astype(np.int)
        
        if len(eventlist) > (length+1):
            start_index = np.random.randint(0, len(eventlist) - (length+1))
            eventlist = eventlist[start_index:start_index+(length+1)]
            
        # pad zeros
        if len(eventlist) < (length+1):
            pad = (length+1) - len(eventlist)
            eventlist = np.pad(eventlist, (pad, 0), 'constant')
            
        x = eventlist[:length]
        y = eventlist[1:length+1] #y就是x的下一步

        # x = np.array([item / EventDim for item in x]) #归一化
        # y = np.array([item / EventDim for item in y])

    elif type == 'poetry':
        trainData = POEMS("../Poetry_Dataset/Poetry_para.txt")
        x, y = trainData.generateBatch()
    
    return x, y
