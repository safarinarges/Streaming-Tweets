

import nltk
from nltk.classify.scikitlearn import SklearnClassifier
import random
import pickle
import io
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

from nltk.tokenize import word_tokenize


class VoteClassifier (ClassifierI):
    def __init__ (self, *classifiers):
        self._classfiers = classifiers
        
    def classify (self, features):
        votes= []
        for c in self._classfiers:
            v = c.classify(features)
            votes.append(v)
        return mode (votes)
    def confidence (self, features):
        votes = []
        for c in self._classfiers:
            v = c.classify (features)
            votes.append(v)
            
        choice_value = votes.count(mode(votes))
        conf = choice_value / len (votes)
        return conf
#short_pos = open ("short_reviews/positive.txt", "r").read()
#short_neg = open ("short_reviews/negative.txt", "r").read()

##io.open(filename, encoding='latin-1')
short_pos = io.open ("short_reviews/positive.txt", encoding='latin-1').read()
short_neg = io.open ("short_reviews/negative.txt", encoding='latin-1').read()

all_words = []
documents = []

#  j is adject, r is adverb, and v is verb
allowed_word_types = ["J"]


for k in short_pos.split('\n'):
    documents.append( (k,"pos") )
    words = word_tokenize(k)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

    
for k in short_neg.split('\n'):
    documents.append( (k, "neg") )
    words = word_tokenize(k)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

save_documents = open("pickled_algorithms/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()

all_words = nltk.FreqDist(all_words)

word_features = list (all_words.keys())[:5000]


save_word_features = open("pickled_algorithms/word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for s in word_features:
        features [s] = (s in words)
        
    return features

 
features_set = [(find_features (rev), category) for (rev, category) in documents]

random.shuffle(features_set)
print(len(features_set))

testing = features_set[10000:]
training = features_set[:10000]

Naiv_classifier = nltk.NaiveBayesClassifier.train(training)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(Naiv_classifier, testing))*100)
Naiv_classifier.show_most_informative_features(15)


#Save the classifier
save_classifier = open("pickled_algorithms/originalnaivebayes5k.pickle.pickle","wb")
pickle.dump(Naiv_classifier, save_classifier)
save_classifier.close()

#classifier_Naiv = open("pickled_algos/naivebayes.pickle","rb")
#Naiv_classifier = pickle.load(classifier_Naiv)
#classifier_Naiv.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing))*100)

save_classifier = open("pickled_algorithms/MNB_classifier5k.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing))*100)

save_classifier = open("pickled_algorithms/BernoulliNB_classifier5k.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing))*100)

save_classifier = open("pickled_algorithms/LogisticRegression_classifier5k.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing))*100)

save_classifier = open("pickled_algorithms/LinearSVC_classifier5k.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()


##NuSVC_classifier = SklearnClassifier(NuSVC())
##NuSVC_classifier.train(training_set)
##print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training)
print("SGDClassifier accuracy percent:",nltk.classify.accuracy(SGDC_classifier, testing)*100)

save_classifier = open("pickled_algorithms/SGDC_classifier5k.pickle","wb")
pickle.dump(SGDC_classifier, save_classifier)
save_classifier.close()





