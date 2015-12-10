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
count=0
with open("C:/Users/MAHATHI/Downloads/trainingandtestdata/training.1600000.processed.noemoticon.csv", "r") as csvfile:
    spamreader = csv.reader(csvfile, dialect='excel')
    for row in spamreader:
        d = enchant.Dict("en_US")
        if(count>2000):
            break
        count = count + 1
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
    negcutoff = int(len(negfeats)*3/4)
    poscutoff = int(len(posfeats)*3/4)
    netcutoff = int(len(netfeats)*3/4)
    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
    print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

    classifier = NaiveBayesClassifier.train(trainfeats)
    f = open('C:/Users/MAHATHI/Downloads/trainingandtestdata/my_classifier.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()
