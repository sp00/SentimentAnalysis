'''
Created on Mar 7, 2012

@author: William
'''

import csv
import nltk
from string import punctuation
from math import log

SMILEYS = (':)', ':d', ':-)', ':(', ':-(', ':p')

def remove_smileys(tweet):
    for w in SMILEYS:
        tweet = tweet.replace(w, '')
    return tweet

def decode(tweet):
    return tweet.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')

def remove_startswith(tweet, startswith):
    begin = tweet.find(startswith)
    
    while begin != -1:
        end = tweet.find(' ', begin)
        if end == -1:
            end = len(tweet)
        
        sub = tweet[begin : end]
        tweet = tweet.replace(sub, '')
        
        begin = tweet.find(startswith)
    
    return tweet

# removes 'rt', '@xxx', '#', 'http://xxx', punctuation, stopwords and words of length 1
# return None if the length of tweet is 0 after 
def remove_tweet_features(tweet):
    purged = tweet.replace('rt', '').replace('#', '')
    purged = remove_startswith(remove_startswith(purged, '@'), 'http://')
    
    stopwords = nltk.corpus.stopwords.words('english')    
    
    words = [word.strip(punctuation) for word in purged.split() \
             if word not in stopwords and len(word) > 1]
    purged = ' '.join(words).strip()
    
    return purged

def purge_tweets(tweets):
    results = []
    
    for tweet in tweets:
        purged = tweet.lower().strip()
        purged = purged.replace('"', '').strip("'")
        purged = remove_smileys(purged)
        purged = remove_tweet_features(decode(purged))
        if purged != '':
            results.append(purged)
    
    return results

def write_hdf(tweets, sentiment):
    f = open('input/%s/purged_tweets_1.txt' % sentiment, 'w')
    
    for i, tweet in enumerate(tweets):
        if i % 10000 == 0:
            f.close()
            f = open('input/%s/purged_tweets_%d.txt' % (sentiment, i / 10000 + 1), 'w')
        
        f.write(tweet + '\n')
    
    f.close()

def process_files():
    with open('face_sentiment.csv') as f:
        reader = csv.DictReader(f)
        pos_t = [row['tweet'] for row in reader if row['sentiment'] == 'positive']
    with open('face_sentiment.csv') as f:
        reader = csv.DictReader(f)
        neg_t = [row['tweet'] for row in reader if row['sentiment'] == 'negative']
        
    with open('face_tweets.csv') as f:
        reader = csv.DictReader(f)
        pos_t += [row['tweet'] for row in reader if row['sentiment'] == 'positive']
    with open('face_tweets.csv') as f:
        reader = csv.DictReader(f)
        neg_t += [row['tweet'] for row in reader if row['sentiment'] == 'negative']
    
    with open('simple_tweets_pos.txt') as f:
        tweets = f.readlines()
    pos_t += tweets
    with open('simple_tweets_neg.txt') as f:
        tweets = f.readlines()
    neg_t += tweets
    
    pos_t = purge_tweets(pos_t)
    neg_t = purge_tweets(neg_t)
    
    # remove duplicates
    pos_t = list(set(pos_t))
    neg_t = list(set(neg_t))
    
    write_hdf(pos_t, 'positive')
    write_hdf(neg_t, 'negative')


# return {'tweet': int(times)}
def retrieve_grams(tweets, n=2):
    grams = {}
    
    for tweet in tweets:
        words = tweet.split()
        
        if len(words) <= n:
            if tweet not in grams:
                grams[tweet] = 1
            else:
                grams[tweet] += 1
            continue
        
        for i in range(len(words) - n + 1):
            gram = ' '.join(words[i : i + n])
            if gram not in grams:
                grams[gram] = 1
            else:
                grams[gram] += 1
    
    return grams


def compute_likelihood(tweet, possibilities, entropy, salient, n=2):
    words = tweet.split()
    grams = []
    
    if len(words) <= n:
        grams = [tweet]
    else:
        for i in range(len(words) - n + 1):
            grams.append(' '.join(words[i : i + n]))
    
    grams = [(gram, possibilities[gram]) for gram in grams \
                if gram in possibilities]
    weight_grams = [(gram, possibilities[gram]) for gram, possibility in grams \
                if gram in entropy]
    
    likelihood = sum(gram[1] for gram in grams)
    weight = sum(gram[1] for gram in weight_grams)
    return likelihood, grams, weight


def primitive_process():
    
    size = 0
    
    with open('positive_face.csv') as f:
        pos_tweets = [row['tweet'] for row in csv.DictReader(f)]
        size = len(pos_tweets)
        print 'Size of all tweets:', size
    pos_grams = retrieve_grams(pos_tweets)
    print 'Size of positive grams:', len(pos_grams)
#    pos_grams = dict(sorted(pos_grams.iteritems(),
#           key=lambda(word, count): (-count, word))[:10000])
    
    with open('negative_face.csv') as f:
        neg_tweets = [row['tweet'] for row in csv.DictReader(f)]
    neg_grams = retrieve_grams(neg_tweets)
    print 'Size of negative grams:', len(neg_grams)
#    neg_grams = dict(sorted(neg_grams.iteritems(),
#           key=lambda(word, count): (-count, word))[:10000])
    
    return pos_grams, neg_grams, size

def load_hadoop_grams():
    pos_grams = {}
    with open('primitive_grams/pos_grams') as f:
        for line in f.readlines():
            parts = line.split('\t')
            pos_grams.update({parts[0]: int(parts[1])})
    print 'Size of positive grams:', len(pos_grams)
            
    neg_grams = {}
    with open('primitive_grams/neg_grams') as f:
        for line in f.readlines():
            parts = line.split('\t')
            neg_grams.update({parts[0]: int(parts[1])})
    print 'Size of negative grams:', len(neg_grams)
    
    return pos_grams, neg_grams

def main():
    
#    pos_grams, neg_grams, size = primitive_process()
    size = 180000
    pos_grams, neg_grams = load_hadoop_grams()
    
    
    # mutual grams
    mutual_grams = []
    for gram in pos_grams:
        if gram in neg_grams:
            mutual_grams.append(gram)
    print 'Size of mutual grams:', len(mutual_grams)
    
    
    
    # compute entropy
    entropy_grams = {}
    for gram in mutual_grams:
        times = pos_grams[gram] + neg_grams[gram]
        # possibility of positive with presence of this gram
        p_pos = float(pos_grams[gram]) / times
        # possibility of negative with presence of this gram
        p_neg = float(neg_grams[gram]) / times
        entropy = - (p_pos * log(p_pos) + p_neg * log(p_neg))
        entropy_grams[gram] = entropy
    
#    # output entropy
#    with open('n_grams_entropy.csv', 'w') as f:
#        writer = csv.writer(f)
#        writer.writerow(['gram', 'entropy'])
#        for gram in entropy_grams:
#            writer.writerow([gram, entropy_grams[gram]])
    entropy_grams = dict(sorted(entropy_grams.iteritems(),
           key=lambda(word, count): (count, word))[:5000])
    
    
    
    # compute possibility of the presence of one gram based on certain sentiment
    # assume each gram appears in one tweet only once
    for gram in pos_grams:
        pos_grams[gram] = float(pos_grams[gram]) / size
    for gram in neg_grams:
        neg_grams[gram] = float(neg_grams[gram]) / size
    
    
    
    # compute salient
    salient_grams = {}
    for gram in mutual_grams:
        salient_grams[gram] = \
            1 - min(pos_grams[gram], neg_grams[gram]) / max(pos_grams[gram], neg_grams[gram])
    
#    # output salient
#    with open('n_grams_salient.csv', 'w') as f:
#        writer = csv.writer(f)
#        writer.writerow(['gram', 'salient'])
#        for gram in salient_grams:
#            writer.writerow([gram, salient_grams[gram]])
    salient_grams = dict(sorted(salient_grams.iteritems(),
           key=lambda(word, count): (-count, word)))
    
    
    
#    # sort entropy ASCE and salient DESC
#    e_top = sorted(entropy_grams.iteritems(),
#           key=lambda(word, count): (count, word))
#    s_top = sorted(salient_grams.iteritems(),
#           key=lambda(word, count): (-count, word))
#    print e_top
#    print s_top
    e_top_grams = set([gram for gram in entropy_grams])
    s_top_grams = set([gram for gram in salient_grams])
    
    with open('n-grams_pos.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['gram', 'possibility'])
        for gram in pos_grams:
            writer.writerow([gram, pos_grams[gram]])
    
    with open('n-grams_neg.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['gram', 'possibility'])
        for gram in neg_grams:
            writer.writerow([gram, neg_grams[gram]])
    
    
    
    with open('test_tweets.csv') as f:
        test_tweets = [(row['sentiment'], row['tweet'].strip()) for row in csv.DictReader(f)]
    test_tweets = list(set(test_tweets))
    
    # TODO: purge test tweets?
    
    total_pos = 0
    total_neg = 0
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    with open('classification.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['l_pos', 'pos_grams', 'pos_weight', 'l_neg', 'neg_grams', 'neg_weight', 'predict', 'sentiment', 'tweet'])
        
        print 'Size of pos_grams:', len(pos_grams)
        print 'Size of neg_grams:', len(neg_grams)
        for tweet in test_tweets:
            l_pos, p_grams, p_weight = compute_likelihood(tweet[1], pos_grams, e_top_grams, s_top_grams)
            l_neg, n_grams, n_weight = compute_likelihood(tweet[1], neg_grams, e_top_grams, s_top_grams)
            
            # thresholds increase precision and non-zeros increase recall
            sentiment = 'neutral'
            if l_pos > l_neg and (l_pos > 0.0005 \
                    or (p_weight > n_weight)) \
                    or (l_pos > 0 and l_neg == 0):
                sentiment = 'positive'
            elif l_neg > l_pos and (l_neg > 0.005 \
                    or (n_weight > p_weight)) \
                    or (l_neg > 0 and l_pos == 0):
                sentiment = 'negative'
            
            writer.writerow([l_pos, p_grams, p_weight, l_neg, n_grams, n_weight, sentiment, tweet[0], tweet[1]])
            
            if tweet[0] == 'positive':
                total_pos += 1
                if sentiment == 'positive':
                    true_pos += 1
                elif sentiment == 'negative':
                    false_neg += 1
            else:
                total_neg += 1
                if sentiment == 'positive':
                    false_pos += 1
                elif sentiment == 'negative':
                    true_neg += 1
    
    print 'Positive - precision: %.1f; recall: %.1f' % \
            (float(true_pos) / (true_pos + false_pos) * 100, \
             float(true_pos) / total_pos * 100)
    print 'Negative - precision: %.1f; recall: %.1f' % \
            (float(true_neg) / (true_neg + false_neg) * 100, \
             float(true_neg) / total_neg * 100)





















