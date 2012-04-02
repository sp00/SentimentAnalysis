#!/opt/local/bin/python -tt

"""
@package sentiment
Twitter sentiment analysis.

This code performs sentiment analysis on Tweets.

A custom feature extractor looks for key words and emoticons.  These are fed in
to a naive Bayes classifier to assign a label of 'positive', 'negative', or
'neutral'.  Optionally, a principle components transform (PCT) is used to lessen
the influence of covariant features.

"""
import csv, random
import nltk
import collections
from nltk import metrics

import tweet_features, tweet_pca

def measure(classifier, testfeats, alpha=0.5):
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    
    for i, (feats, label) in enumerate(testfeats):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)
    
    precisions = {}
    recalls = {}
    f_measures = {}
    for label in classifier.labels():
        precisions[label] = metrics.precision(refsets[label], testsets[label])
        recalls[label] = metrics.recall(refsets[label], testsets[label])
        f_measures[label] = metrics.f_measure(refsets[label], testsets[label], alpha)
	
    return precisions, recalls, f_measures

def cross_validate(corpus, fold, alpha=0.5):
    split = len(corpus) / fold
    ave_accuracy = 0
    ave_precision = {'positive': 0, 'negative': 0, 'neutral': 0}
    ave_recall = {'positive': 0, 'negative': 0, 'neutral': 0}
    ave_f_measure = {'positive': 0, 'negative': 0, 'neutral': 0}
    
    for i in range(fold):
        print '\nMetrics in run %d:' % (i + 1)
        
        v_test = corpus[split * i:split * (i + 1)]
        v_train = corpus[:split * i] + corpus[split * (i + 1):]
        classifier = nltk.NaiveBayesClassifier.train(v_train);
        
        accuracy = nltk.classify.accuracy(classifier, v_test)
        print 'Accuracy: %f' % accuracy
        ave_accuracy += accuracy
        
        precisions, recalls, f_measures = measure(classifier, v_test, alpha)
        print 'Precisions:', precisions
        print 'Recalls:', recalls
        print 'F-Measures:', f_measures
        for label in precisions:
            ave_precision[label] += precisions[label]
        for label in recalls:
            ave_recall[label] += recalls[label]
        for label in f_measures:
            ave_f_measure[label] += f_measures[label]
        
    print '\nMetrics after %d-fold cross validation:' % fold
    print 'Avarage accuracy:', ave_accuracy / fold
    for label in ave_precision:
        ave_precision[label] /= fold
    for label in ave_recall:
        ave_recall[label] /= fold
    for label in ave_f_measure:
        ave_f_measure[label] /= fold
    print 'Avarage precision:', ave_precision	
    print 'Avarage recall:', ave_recall	
    print 'Avarage f_measure:', ave_f_measure
    
def basic_analyze(fvecs):
    # split in to training and test sets
    v_train = fvecs[:2500]
    v_test  = fvecs[2500:]

    # train classifier
    nb_classifier = nltk.NaiveBayesClassifier.train(v_train)
    #classifier = nltk.classify.maxent.train_maxent_classifier_with_gis(v_train);
    me_classifier = nltk.MaxentClassifier.train(v_train, algorithm='iis', trace=0, max_iter=1, min_lldelta=0.5)

    # classify and dump results for interpretation
    print '\nNaiveBayesClassifier Accuracy %f' % nltk.classify.accuracy(nb_classifier, v_test)
    print '\nMaxentClassifier Accuracy %f\n' % nltk.classify.accuracy(me_classifier, v_test)
    #print classifier.show_most_informative_features(200)

    precisions, recalls, f_measures = measure(nb_classifier, v_test, 0.8)
    print 'NB_Precisions:', precisions
    print 'NB_Recalls:', recalls
    print 'NB_F-Measures:', f_measures
    
    precisions, recalls, f_measures = measure(me_classifier, v_test, 0.8)
    print 'ME_Precisions:', precisions
    print 'ME_Recalls:', recalls
    print 'ME_F-Measures:', f_measures

    # build confusion matrix over test set
    test_truth   = [s for (t,s) in v_test]
    test_predict = [nb_classifier.classify(t) for (t,s) in v_test]

    print '\nConfusion Matrix'
    print nltk.ConfusionMatrix( test_truth, test_predict )

def main():
# read all tweets and labels
    fp = open( 'sentiment.csv', 'rb' )
    reader = csv.reader( fp, delimiter=',', quotechar='"', escapechar='\\' )
    tweets = []
    for row in reader:
        tweets.append( [row[4], row[1]] )
    
    fp.close()

    # treat neutral and irrelevant the same
    for t in tweets:
        if t[1] == 'irrelevant':
            t[1] = 'neutral'

    random.shuffle( tweets );
    fvecs = [(tweet_features.make_tweet_dict(t),s) for (t,s) in tweets]

    # dump tweets which our feature selector found nothing
    #for i in range(0,len(tweets)):
    #    if tweet_features.is_zero_dict( fvecs[i][0] ):
    #        print tweets[i][1] + ': ' + tweets[i][0]

    # apply PCA reduction
    #(v_train, v_test) = \
    #        tweet_pca.tweet_pca_reduce( v_train, v_test, output_dim=1.0 )
    
    basic_analyze(fvecs)

    cross_validate(fvecs, 10, 0.8)
    
if __name__ == '__main__':
    main()
