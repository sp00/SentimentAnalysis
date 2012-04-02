import collections
import csv, random
import nltk
from nltk import metrics
import tweet_features, tweet_pca
from math import log
from count_freq import count_freq
from collections import defaultdict

MI = "MUTUAL_INFORMATION"
CHI = 'CHI_SQUARE'

def mi_formula(n11, n10, n01, n00):
    n = n11+n10+n01+n00
    n1x = n10+n11
    nx1 = n01+n11
    n0x = n00+n01
    nx0 = n10+n00
    a = n11*log(n*n11/(n1x*nx1), 2) if n11 > 0 else 0
    b = n01*log(n*n01/(n0x*nx1), 2) if n01 > 0 else 0
    c = n10*log(n*n10/(n1x*nx0), 2) if n10 > 0 else 0
    d = n00*log(n*n00/(n0x*nx0), 2) if n00 > 0 else 0
    return (a + b + c + d) / n

def chi_formula(n11, n10, n01, n00):
    n = n11+n10+n01+n00;
    n1x = n10+n11;
    nx1 = n01+n11;
    n0x = n00+n01;
    nx0 = n10+n00;
    return n*(n11*n00-n10*n01)**2/(nx1*n1x*n0x*nx0);

def mi_oneword(word, category, stat):
    n11=0.0;n10=0.0;n00=0.0;n01=0.0;
    reader = csv.reader( open("sentiment.csv","rb"), delimiter=',', quotechar='"', escapechar='\\' )
    for row in reader:
        if(word in row[4]):
            if(row[1]==category):
                n11=n11+1;
            else:
                n10=n10+1;
        else:
            if(row[1]==category):
                n01=n01+1;
            else:
                n00=n00+1;
    #0:mutual information, #1: measure
    if(stat == MI):   
        # if(n11 == 0):
        #     return -11;
        # elif(n10 == 0):
        #     return -10;
        # elif(n00 == 0):
        #     return 0;
        # elif(n01 == 0):
        #     return -1;
        # else:    
        return mi_formula(n11, n10, n01, n00);
    elif(stat == CHI):
        # if(n11+n10==0):
        #     return -12;
        # elif(n11+n01==0):
        #     return -21;
        # elif(n01+n00==0):
        #     return -2;
        # elif(n10+n00==0):
        #     return -20;
        # else:
        return chi_formula(n11, n10, n01, n00);
        
def feature_selection(category, frequency, stat):

    top_words = count_freq('sentiment.csv', frequency)
    print '\n\n\n'
    print top_words
    print '\n\n\n'
    
    mi_result = {}
    mi_result = defaultdict(int)
    for word, frequency in top_words:
        result = mi_oneword(word, category, stat)
        if result > 0:
            mi_result[word] = result
            # if(stat==MI):
            #     if(mi != -11 and mi != -10 and mi != 0 and mi != -1):
            #         mi_result[word] = mi;
            # elif(stat==CHI):
            #     if(mi !=-12 and mi != -21 and mi != -2 and mi != -20):
            #         mi_result[word] = mi;
            print word, ':', mi_result[word] 
    #print mi_oneword('the', 'positive')
    feature = {}
    feature = sorted(mi_result.iteritems(), key=lambda(word, count) : (-count, word))[:len(mi_result) - 1]
    return feature

def dumpTwittes(in_file_name, out_file_name, sentiment):
    reader = csv.reader(open(in_file_name, 'r'), delimiter=',', quotechar='"')
    writer = csv.writer(open(out_file_name, 'a'), delimiter=',', quotechar='"')
    for row in reader:
        if row[1] == sentiment:
            writer.writerow(row)

def frequency_selector():
    dumpTwittes('sentiment.csv', 'positive_twittes_dump.csv', 'positive')
    dumpTwittes('sentiment.csv', 'negative_twittes_dump.csv', 'negative')
    dumpTwittes('sentiment.csv', 'neutral_twittes_dump.csv', 'neutral')
    dumpTwittes('sentiment.csv', 'neutral_twittes_dump.csv', 'irrelevant')
    
    positive_words = dict(count_freq('positive_twittes_dump.csv', 1000))
    negative_words = dict(count_freq('negative_twittes_dump.csv', 1000))
    neutral_words = dict(count_freq('neutral_twittes_dump.csv', 1000))
    
    popular_control = 50
    threshold = 100
    
    positive_features = {}
    for word in positive_words:
        if (word in negative_words and (negative_words[word] > popular_control or positive_words[word] < negative_words[word] * threshold)) \
            or (word in neutral_words and (neutral_words[word] > popular_control or positive_words[word] < neutral_words[word] * threshold)):
            continue
        positive_features[word] = positive_words[word]
    positive_features = sorted(positive_features.iteritems(),
                   key=lambda(word, count): (-count, word))[:100] 
#    print positive_features
    
    negative_features = {}
    for word in negative_words:
        if (word in positive_words and (positive_words[word] > popular_control or negative_words[word] < positive_words[word] * threshold)) \
            or (word in neutral_words and (neutral_words[word] > popular_control or negative_words[word] < neutral_words[word] * threshold)):
            continue
        negative_features[word] = negative_words[word]
    negative_features = sorted(negative_features.iteritems(),
                   key=lambda(word, count): (-count, word))[:100]
    
    neutral_features = {}
    for word in neutral_words:
        if (word in negative_words and (negative_words[word] > popular_control or neutral_words[word] < negative_words[word] * threshold)) \
            or (word in positive_words and (positive_words[word] > popular_control or neutral_words[word] < positive_words[word] * threshold)):
            continue
        neutral_features[word] = neutral_words[word]
    neutral_features = sorted(neutral_features.iteritems(),
                   key=lambda(word, count): (-count, word))[:100]
    
    spamWriter = csv.writer(open('postive_features.csv', 'w'), delimiter=',', quotechar='"')
    spamWriter.writerow(['***postive***', '***MI***'])
    for word, freq in positive_features:
        spamWriter.writerow([word, freq])
    spamWriter = csv.writer(open('negative_features.csv', 'w'), delimiter=',', quotechar='"')
    spamWriter.writerow(['***negative***', '***MI***'])
    for word, freq in negative_features:
        spamWriter.writerow([word, freq])
    spamWriter = csv.writer(open('neutral_features.csv', 'w'), delimiter=',', quotechar='"')
    spamWriter.writerow(['***neutral***', '***MI***'])
    for word, freq in neutral_features:
        spamWriter.writerow([word, freq])

frequency_selector()

##spamWriter.writerow(['positive', 'MI', 'negive','MI' ])
#print 'positive analysis start:\n'
#positive_feature = dict(feature_selection('positive', 10000, MI))
#print 'negative analysis start:\n'# 
#negative_feature = dict(feature_selection('negative', 10000, MI))
#print 'neutral analysis start:\n'
#neutral_feature = dict(feature_selection('neutral', 10000, MI))
#
#threshold = 50
#
#positive_features = {}
#for key in positive_feature.keys():
#    if negative_feature.has_key(key) and negative_feature[key] * threshold > positive_feature[key] or \
#        neutral_feature.has_key(key) and neutral_feature[key] * threshold > positive_feature[key]:
#        continue
#    positive_features[key] = positive_feature[key]
#
#negative_features = {}
#for key in negative_feature.keys():
#    if positive_feature.has_key(key) and positive_feature[key] * threshold > negative_feature[key] or \
#        neutral_feature.has_key(key) and neutral_feature[key] * threshold > negative_feature[key]:
#        continue
#    negative_features[key] = negative_feature[key]
#    
#neutral_features = {}
#for key in neutral_feature.keys():
#    if positive_feature.has_key(key) and positive_feature[key] * threshold > neutral_feature[key] or \
#        negative_feature.has_key(key) and negative_feature[key] * threshold > neutral_feature[key]:
#        continue
#    neutral_features[key] = neutral_feature[key]
#
#spamWriter = csv.writer(open('postive_features.csv', 'w'), delimiter=',', quotechar='"')
#spamWriter.writerow(['***postive***', '***MI***'])
#for word in positive_features.keys():
#    spamWriter.writerow([word, positive_features[word]])
#spamWriter = csv.writer(open('negative_features.csv', 'w'), delimiter=',', quotechar='"')
#spamWriter.writerow(['***negative***', '***MI***'])
#for word in negative_features.keys():
#    spamWriter.writerow([word, negative_features[word]])
#spamWriter = csv.writer(open('neutral_features.csv', 'w'), delimiter=',', quotechar='"')
#spamWriter.writerow(['***neutral***', '***MI***'])
#for word in neutral_features.keys():
#    spamWriter.writerow([word, neutral_features[word]])



    
