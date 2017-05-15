#!/usr/bin/python3

import nltk
import sys
import json
from sys import exit

#pos_comments = [('I love this car', 'positive'),
#              ('This view is amazing', 'positive'),
#              ('I feel great this morning', 'positive'),
#              ('I am so excited about the concert', 'positive'),
#              ('He is my best friend', 'positive'),
#              ('Going well', 'positive'),
#              ('Thank you', 'positive'),
#              ('Hope you are doing well', 'positive'),
#              ('I am very happy', 'positive'),
#              ('Good for you', 'positive'),
#              ('It is all good. I know about it and I accept it.', 'positive'),
#              ('This is really good!', 'positive'),
#              ('Tomorrow is going to be fun.', 'positive'),
#              ('Smiling all around.', 'positive'),
#              ('These are great apples today.', 'positive'),
#              ('How about them apples? Thomas is a happy boy.', 'positive'),
#              ('Thomas is very zen. He is well-mannered.', 'positive')]

#neg_comments = [('I do not like this car', 'negative'),
#              ('This view is horrible', 'negative'),
#              ('I feel tired this morning', 'negative'),
#              ('I am not looking forward to the concert', 'negative'),
#              ('He is my enemy', 'negative'),
#              ('I am a bad boy', 'negative'),
#              ('This is not good', 'negative'),
#              ('I am bothered by this', 'negative'),
#              ('I am not connected with this', 'negative'),
#              ('Sadistic creep you ass. Die.', 'negative'),
#              ('All sorts of crazy and scary as hell.', 'negative'),
#              ('Not his emails, no.', 'negative'),
#              ('His father is dead. Returned obviously.', 'negative'),
#              ('He has a bomb.', 'negative'),
#              ('Too fast to be on foot. We cannot catch them.', 'negative')]

pos_comments=[]
with open('pos.json','r') as f:
    pos_comments=json.load(f)

neg_comments=[]
with open('neg.json','r') as f:
    neg_comments=json.load(f)

comments = []
for (words, sentiment) in pos_comments + neg_comments:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
    comments.append((words_filtered, sentiment))

def get_words_in_comments(comments):
    all_words = []
    for (words, sentiment) in comments:
      all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

word_features = get_word_features(get_words_in_comments(comments))

training_set = nltk.classify.apply_features(extract_features, comments)
classifier = nltk.NaiveBayesClassifier.train(training_set)

save_classifier = open("tweetposneg.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

# classifier_f = open("naivebayes.pickle", "rb")
# classifier = pickle.load(classifier_f)
# classifier_f.close()

#runcomments = []
#if len(sys.argv) > 1:
#    tweetfile = sys.argv[1]
#    with open(tweetfile, "r") as ins:
#      for line in ins:
#        runcomments.append(line)
#runcomments.append('I am a bad boy')
with open('test.json','r') as f:
    jsoncomments=json.load(f)

poscount = 0
negcount = 0
for commentt in runcomments:
  valued = classifier.classify(extract_features(commentt.split()))
  print (valued)
  if valued == 'negative':
    negcount = negcount + 1
  else:
    poscount = poscount + 1
    print ('Positive count: %s \nNegative count: %s' % (poscount,negcount))
  exit()