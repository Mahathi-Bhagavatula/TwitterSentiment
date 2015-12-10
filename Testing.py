import csv
import pickle
import enchant
import nltk.classify.util
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
with open("testdata.manual.2009.06.14.csv", "r") as csvfile:
    spamreader = csv.reader(csvfile, dialect='excel')
    for row in spamreader:
        d = enchant.Dict("en_US")
        count = count + 1
        cleanedString = newPreprocess(row[5])
        englishString = removeNonEnglishWords(cleanedString,d)
        if(row[0]=='0'):
            negids.append(englishString.split())
        elif(row[0]=='2'):
            netids.append(englishString.split())
        elif(row[0]=='4'):
            posids.append(englishString.split())
    print(count)
    negfeats = [(word_feats(f), 'neg') for f in negids]
    posfeats = [(word_feats(f), 'pos') for f in posids]
    netfeats = [(word_feats(f), 'net') for f in netids]
    testfeats = negfeats[:] + posfeats[:]
    print('train on %d instances, test on %d instances' % (0, len(testfeats)))
    f = open('my_classifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()
    print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
    classifier.show_most_informative_features()
