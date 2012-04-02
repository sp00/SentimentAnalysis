'''
Created on Mar 15, 2012

@author: William
'''

import sys
import csv
import nltk
from string import punctuation
from math import log


F_NAME_POS = 'n-grams_pos.csv'
F_NAME_NEG = 'n-grams_neg.csv'
F_NAME_WEIGHT = 'n_grams_weight.csv'
F_NAME_RESULTS = 'results.txt'

POS_SMILEYS = (':)', ':d', ':-)', ':p')
NEG_SMILEYS = (':(', ':-(')
SMILEYS = POS_SMILEYS + NEG_SMILEYS

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
        purged = tweet[1].lower().strip()
        purged = purged.replace('"', '').strip("'")
        purged = remove_smileys(purged)
        purged = remove_tweet_features(decode(purged))
        results.append((tweet[0], purged))
    
    return results


# return a list: [ ('id', 'tweet') ]
def load_tweets(file_name):
    with open(file_name) as f:
        reader = csv.reader(f)
        return [(row[0], row[1]) for row in reader]


def load_grams():
    
    with open(F_NAME_POS) as f:
        reader = csv.DictReader(f)
        pos_grams = dict([(row['gram'], float(row['possibility'])) for row in reader])
    
    with open(F_NAME_NEG) as f:
        reader = csv.DictReader(f)
        neg_grams = dict([(row['gram'], float(row['possibility'])) for row in reader])
    
    with open(F_NAME_WEIGHT) as f:
        reader = csv.DictReader(f)
        weight_grams = dict([(row['gram'], float(row['possibility'])) for row in reader])
    
    return pos_grams, neg_grams, weight_grams


def score(tweet, possibilities, weight_grams, n=2):
    words = tweet.split()
    grams = []
    
    if len(words) <= n:
        grams = [tweet]
    else:
        for i in range(len(words) - n + 1):
            grams.append(' '.join(words[i : i + n]))
    
    grams = [(gram, possibilities[gram]) for gram in grams \
                if gram in possibilities]
    w_grams = [(gram, possibilities[gram]) for gram, possibility in grams \
                if gram in weight_grams]
    
    t_score = sum(gram[1] for gram in grams)
    senti_weight = sum(gram[1] for gram in w_grams)
#    print t_score, senti_weight, tweet
    return t_score, senti_weight, grams


def output(results):
            
    pos_n = 0
    neg_n = 0
    neu_n = 0
    
#    with open('tmp.csv', 'w') as f:
#        writer = csv.writer(f)
#        for row in results:
#            writer.writerow(row)
    
    with open(F_NAME_RESULTS, 'w') as f:
        
        for row in results:
            if row[6] == 'positive':
                pos_n += 1
                sentiment = 1
                possibility = row[0] / (row[0] + row[3])
            elif row[6] == 'negative':
                neg_n += 1
                sentiment = -1
                possibility = row[3] / (row[0] + row[3])
            else:
                neu_n += 1
                sentiment = 0
                possibility = 0
            
            f.write('%s %d %.2f\n' % (row[7], sentiment, possibility))
            
    total = pos_n + neg_n + neu_n 
    print 'Total: %d; positive: %d [%.2f]; negative: %d [%.2f]; neutral: %d [%.2f].' % \
        (total, pos_n, float(pos_n) / total * 100, neg_n, float(neg_n) / total * 100, neu_n, float(neu_n) / total * 100)


def classify(tweets, pos_grams, neg_grams, weight_grams, p_threshold, n_threshold):
    
    rows = []
    for tweet in tweets:
        
        p_likelihood = 0
        n_likelihood = 0
        p_weight = 0
        n_weight = 0
        p_grams = {}
        n_grams = {}
        sentiment = 'neutral'
        t = tweet[1]
        
        if t != '':
            is_pos = False
            is_neg = False
            for pos_smiley in POS_SMILEYS:
                if t.find(pos_smiley) != -1:
                    is_pos = True
                    p_likelihood = 1
                    sentiment = 'positive'
            for neg_smiley in NEG_SMILEYS:
                if t.find(neg_smiley) != -1:
                    is_neg = True
                    n_likelihood = 1
                    sentiment = 'negative'
            
            # shortcut if the tweet contains smileys
            if is_pos or is_neg:
                if is_pos and is_neg:
                    sentiment = 'neutral'
            else:
                p_likelihood, p_weight, p_grams = score(tweet[1], pos_grams, weight_grams)
                n_likelihood, n_weight, n_grams = score(tweet[1], neg_grams, weight_grams)
                
                # thresholds increase precision and non-zeros increase recall
                if p_likelihood > n_likelihood and (p_likelihood > p_threshold \
                        or (p_weight > n_weight)) \
                        or (p_likelihood > 0 and n_likelihood == 0):
                    sentiment = 'positive'
                elif n_likelihood > p_likelihood and (n_likelihood > n_threshold \
                        or (n_weight > p_weight)) \
                        or (n_likelihood > 0 and p_likelihood == 0):
                    sentiment = 'negative'
        
        row = [p_likelihood, p_weight, p_grams, n_likelihood, n_weight, n_grams, sentiment, tweet[0], tweet[1]]
        rows.append(row)
        
    output(rows)


def main(file_name):
    tweets = purge_tweets(load_tweets(file_name))
    
    pos_grams, neg_grams, weight_grams = load_grams()
    print 'Size of pos_grams:', len(pos_grams)
    print 'Size of neg_grams:', len(neg_grams)
    
    weight_grams = dict(sorted(weight_grams.iteritems(),
           key=lambda(word, count): (count, word))[:50000])
    print 'Size of weight_grams:', len(weight_grams)
    
    classify(tweets, pos_grams, neg_grams, weight_grams, 0.00001, 0.00001)


if __name__ == '__main__':
    main(sys.argv[1])





