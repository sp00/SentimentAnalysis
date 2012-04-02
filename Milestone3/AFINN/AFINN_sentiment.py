#!/opt/local/bin/python -tt

import csv
import nltk

afinn = dict(map(lambda (k,v): (k,int(v)), 
                     [ line.split('\t') for line in open("AFINN-111.txt") ]))

def calculate(threshold):
    
    err_counter = 0
    total_pos = 0
    total_neg = 0
    total_neu = 0
    predict_pos = 0
    predict_neg = 0
    predict_neu = 0
    true_pos = 0
    true_neg = 0
    true_neu = 0
    v_truth = []
    v_predict = []
    highest_score = 0
    lowest_score = 0

    reader = csv.reader(open('sentiment.csv', 'r'), delimiter=',', quotechar='"', escapechar='\\')
    writer = csv.writer(open('sentiment_incorret_%s.csv' % threshold, 'w'), delimiter=',', quotechar='"', escapechar='\\')
    
    for row in reader:
        v_truth.append(row[1] if row[1] != 'irrelevant' else 'neutral')
        if row[1] == 'positive':
            total_pos += 1
        elif row[1] == 'negative':
            total_neg += 1
        else:
            total_neu += 1
        
        score = sum(map(lambda word: afinn.get(word, 0), row[4].lower().split()))
        if score > threshold:
            predict_pos += 1
            v_predict.append('positive')
            highest_score = score if score > highest_score else highest_score
            if row[1] != 'positive':
                writer.writerow([row[1], 'positive', score, row[4]])
                err_counter += 1
            else:
                true_pos += 1
        elif score < -threshold:
            predict_neg += 1
            v_predict.append('negative')
            lowest_score = score if score < lowest_score else lowest_score
            if row[1] != 'negative':
                writer.writerow([row[1], 'negative', score, row[4]])
                err_counter += 1
            else:
                true_neg += 1
        else:
            predict_neu += 1
            v_predict.append('neutral')
            if row[1] == 'positive' or row[1] == 'negative':
                writer.writerow([row[1], 'neutral', score, row[4]])
                err_counter += 1
            else:
                true_neu += 1
    
    print '\nHighest score:', highest_score, '; lowest score:', lowest_score, '\n'
    print 'Number of incorrect classifications:', err_counter, '\n'
    print nltk.ConfusionMatrix( v_truth, v_predict )
    
    pos_p = float(true_pos) / predict_pos
    pos_r = float(true_pos) / total_pos
    pos_f = 2 * pos_p * pos_r / (pos_p + pos_r)
    print '\nPositive: precision -', pos_p, '; recall -', pos_r, '; f-measure -', pos_f
    
    neg_p = float(true_neg) / predict_neg
    neg_r = float(true_neg) / total_neg
    neg_f = 2 * neg_p * neg_r / (neg_p + neg_r)
    print '\Negative: precision -', neg_p, '; recall -', neg_r, '; f-measure -', neg_f
    
    neu_p = float(true_neu) / predict_neu
    neu_r = float(true_neu) / total_neu
    neu_f = 2 * neu_p * neu_r / (neu_p + neu_r)
    print '\Neutral: precision -', neu_p, '; recall -', neu_r, '; f-measure -', neu_f
    
    avg_p = (pos_p + neg_p + neu_p) / 3
    avg_r = (pos_r + neg_r + neu_r) / 3
    avg_f = (pos_f + neg_f + neu_f) / 3
    print '\nAverage: precision -', avg_p, '; recall -', avg_r, '; f-measure -', avg_f, '\n'
    
    return avg_p, avg_r, avg_f


if __name__ == '__main__':
    results = {}
    for i in range (10):
        print '\n=================== threshold', i, '==================='
        results[i] = calculate(i)
    
    print '\n================================================'
    for i in range(10):
        print 'Threshold', i, ':', map(lambda v: '%.2f' % v, results[i])
    print '================================================\n'


