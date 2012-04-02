#!/opt/local/bin/python -tt

import csv
from math import log

def load_tweets():
    tweets_reader = csv.reader(open('sentiment.csv', 'r'), delimiter=',', quotechar='"', escapechar='\\')
    tweets = {}
    all_in_one = ''
    for row in tweets_reader:
        tweets[row[4]] = row[1]
        all_in_one += row[4]
    
    return tweets, all_in_one

def compute_wf():
    tweets, all_in_one = load_tweets()
    feature_reader = csv.reader(open('features.csv', 'r'), delimiter=',', quotechar='"', escapechar='\\')
    
    wf = {}
    for row in feature_reader:
        term = row[0]
        term_cnt = all_in_one.count(term)
        if term_cnt > 0:
            wf[term] = (eval(row[1]), eval(row[2]), log(float(len(tweets)) / term_cnt))
    
    return tweets, wf

def main():
    tweets, wf = compute_wf()
    
    for tweet in tweets:
        words = tweet.split()
        score = 0
        terms = {}
        for term in wf:
            term_cnt = words.count(term)
            if term_cnt != 0:
                terms[term] = term_cnt
                score += (1 + log(term_cnt)) * wf[term][0] * wf[term][1] * wf[term][2]
        sentiment = ''
        tweets[tweet] = (tweets[tweet], score, sentiment, terms)
    
    result_writer = csv.writer(open('results.csv', 'w'), delimiter=',', quotechar='"', escapechar='\\')
    for tweet in tweets:
        result_writer.writerow([tweet, tweets[tweet][0], tweets[tweet][1], tweets[tweet][2], tweets[tweet][3]])

if __name__ == '__main__':
    main()



