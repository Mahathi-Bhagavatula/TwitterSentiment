import csv
import enchant
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
import pickle
def newPreprocess(string):
    chars = "!@#$%^&*(){}[]:;""''<>,./?|\\1234567890"
    for c in chars:
        string = string.replace(c,"")
    return string
def removeNonEnglishWords(string,d):
    ret=""
    splits = string.split()
    for each in splits:
        if d.check(each):
            ret = ret+" "+each
    return ret
def word_feats(words):
    return dict([(word, True) for word in words])
classVariables={}
negids=[]
posids=[]
netids=[]
with open("training.1600000.processed.noemoticon.csv", "r") as csvfile:
    d = enchant.Dict("en_US")
    spamreader = csv.reader(csvfile, dialect='excel')
    for row in spamreader:
        cleanedString = newPreprocess(row[5])
        englishString = removeNonEnglishWords(cleanedString,d)
        if(row[0]=='0'):
            negids.append(englishString.split())
        elif(row[0]=='2'):
            netids.append(englishString.split())
        elif(row[0]=='4'):
            posids.append(englishString.split())
    negfeats = [(word_feats(f), 'neg') for f in negids]
    posfeats = [(word_feats(f), 'pos') for f in posids]
    netfeats = [(word_feats(f), 'net') for f in netids]
    trainfeats = negfeats[:] + posfeats[:]
    classifier = NaiveBayesClassifier.train(trainfeats)
    f = open('my_classifier.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()
