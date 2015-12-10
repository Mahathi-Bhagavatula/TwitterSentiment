import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import words
import enchant
#help(movie_reviews)
negids = movie_reviews.fileids('neg')
#help(negids)
def word_feats(words):
    return dict([(word, True) for word in words])
for f in negids:
    print(f)
    print((movie_reviews.words(fileids=[f])))
negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
#print(negfeats)
#print(type(negfeats))
if "paid" in words.words():
    print("Yes")

def removeNonEnglishWords(string,d):
    ret=""
    splits = string.split()
    for each in splits:
        if d.check(each):
            print("YES")
    return ret
d = enchant.Dict("en_US")
removeNonEnglishWords("paid",d)
