#!/opt/local/bin/python -tt

import csv

weight_magnifier = 1000

# adjective
obj_weight_threshold = .003 # if obj_weight is greater than this value, treat the tweet as neutral regardless of its score
pos_score_threshold = .07   # the score of a tweet must be greater than this value in order to be treated as positive
neg_score_threshold = -.001 # the score of a tweet must be less than this value in order to be treated as negative

## adjective + adverb
#obj_weight_threshold = .005 # if obj_weight is greater than this value, treat the tweet as neutral regardless of its score
#pos_score_threshold = .1   # the score of a tweet must be greater than this value in order to be treated as positive
#neg_score_threshold = -.01 # the score of a tweet must be less than this value in order to be treated as negative

#max_pos_p: ['0.30', '0.11', '0.16'] -- [obj_weight_threshold: 0.06, pos_score_threshold: 0.09, neg_weigh_threshold: -0.01]
#max_neg_p: ['0.32', '0.04', '0.07'] -- [obj_weight_threshold: 0.12, pos_score_threshold: 0.08, neg_weigh_threshold: -0.16]
#max_neu_p: ['0.82', '0.87', '0.84'] -- [obj_weight_threshold: 0.17, pos_score_threshold: 0.08, neg_weigh_threshold: -0.01]

def dispatch_pos():
    reader = csv.reader( open('SentiWordNet.csv', 'r'), delimiter=',', quotechar='"', escapechar='\\' )
    a_writer = csv.writer( open('senti_a.csv', 'w'), delimiter=',', quotechar='"', escapechar='\\' )
    n_writer = csv.writer( open('senti_n.csv', 'w'), delimiter=',', quotechar='"', escapechar='\\' )
    r_writer = csv.writer( open('senti_r.csv', 'w'), delimiter=',', quotechar='"', escapechar='\\' )
    v_writer = csv.writer( open('senti_v.csv', 'w'), delimiter=',', quotechar='"', escapechar='\\' )
    
    for row in reader:
        if row[0] == 'a':
            a_writer.writerow(row)
        elif row[0] == 'n':
            n_writer.writerow(row)
        elif row[0] == 'r':
            r_writer.writerow(row)
        elif row[0] == 'v':
            v_writer.writerow(row)

# {word: [(sentiment, value)]} -- sentiment: 1 - pos, -1 - neg, 0 - obj
def map_words(file_name):
    reader = csv.reader( open(file_name, 'r'), delimiter=',', quotechar='"', escapechar='\\' )
    sense_counter = 0
    all_words = {}
    
    for row in reader:
        sense_counter += 1
        synsets = row[4].split()
        
        for word in synsets:
            word = word.split('#')[0]
            if not word in all_words: all_words[word] = []
            
            if eval(row[2]) == 0 and eval(row[3]) == 0:
                posting = (0, 1)
                all_words[word].append(posting)
            else:
                if row[2] >= row[3]:
                    posting = (1, eval(row[2]))
                    all_words[word].append(posting)
                if row[3] >= row[2]:
                    posting = (-1, eval(row[3]))
                    all_words[word].append(posting)
    
    print '\nTotal sense #:', sense_counter, '; total words #:', len(all_words)
    return all_words, sense_counter

# {word: {'pos_socre': score, 'neg_score': score, 'post_weight': weight, 'neg_weight': weight, 'obj_weight': weight}}
def reduce_words(all_words, sense_counter):
    
    for word in all_words:
        metrics = {}
        
        pos_scores = [posting[1] for posting in all_words[word] if posting[0] == 1]
        metrics['pos_score'] = sum(pos_scores) / len(pos_scores) if len(pos_scores) > 0 else 0
        neg_scores = [posting[1] for posting in all_words[word] if posting[0] == -1]
        metrics['neg_score'] = sum(neg_scores) / len(neg_scores) if len(neg_scores) > 0 else 0
        
        metrics['pos_weight'] = float(len([posting for posting in all_words[word] if posting[0] == 1])) / sense_counter * weight_magnifier
        metrics['neg_weight'] = float(len([posting for posting in all_words[word] if posting[0] == -1])) / sense_counter * weight_magnifier
        metrics['obj_weight'] = float(len([posting for posting in all_words[word] if posting[0] == 0])) / sense_counter * weight_magnifier
        
        all_words[word] = metrics

# {word: (sentiment, score, weight, {<metrics>})}
def assign_sentiment(all_words):
    for word, postings in all_words.items():
        if postings['obj_weight'] > postings['pos_weight'] and postings['obj_weight'] > postings['neg_weight']: # or controlled via certain value?
            senti = (0, 1, postings['obj_weight'], postings)
        elif postings['pos_score'] >= postings['neg_score'] and postings['pos_weight'] > postings['neg_weight'] or \
            (postings['pos_score'] > postings['neg_score'] and postings['pos_weight'] == postings['neg_weight']):
            senti = (1, postings['pos_score'], postings['pos_weight'], postings)
        elif postings['neg_score'] >= postings['pos_score'] and postings['neg_weight'] > postings['pos_weight'] or \
            (postings['neg_score'] > postings['pos_score'] and postings['neg_weight'] == postings['pos_weight']):
            senti = (-1, postings['neg_score'], postings['neg_weight'], postings)
        elif postings['neg_score'] == postings['pos_score'] and postings['neg_weight'] == postings['pos_weight']:
            senti = ('random', 0, 0, postings)
        elif postings['pos_score'] > postings['neg_score']:
            senti = (1, postings['pos_score'], postings['pos_weight'], postings)
        else:
            senti = (-1, postings['neg_score'], postings['neg_weight'], postings)
        
        all_words[word] = senti

# {word: (sentiment, score, weight)}
def extract_features(all_words):
    features = {}
    cnt_features = 0
    for word in all_words:
        if all_words[word][0] != 'random':
            cnt_features += 1
            features[word] = all_words[word][:-1]
    print '\nFeature words #:', cnt_features
    return features

def analyze(tweet, features):
    senti_score = 0
    obj_weight = 0
    
    words = {}
    for word in tweet.split():
        if word in features:
            words[word] = features[word]
            senti_score += features[word][0] * features[word][1] * features[word][2]
            obj_weight += features[word][2] if features[word][0] == 0 else 0
    
#    print '\n', words
    return senti_score, obj_weight, words

def classifier(file_name, features, obj_weight_threshold, pos_score_threshold, neg_score_threshold):
    reader = csv.reader(open(file_name, 'r'), delimiter=',', quotechar='"', escapechar='\\' )
#    pos_writer = csv.writer(open('results/%s_%s_%s_false_positive.csv' \
#                                  % (obj_weight_threshold, pos_score_threshold, neg_score_threshold), 'w'), \
#                            delimiter=',', quotechar='"', escapechar='\\')
#    neg_writer = csv.writer(open('results/%s_%s_%s_false_negative.csv' \
#                                  % (obj_weight_threshold, pos_score_threshold, neg_score_threshold), 'w'),\
#                            delimiter=',', quotechar='"', escapechar='\\')
#    neu_writer = csv.writer(open('results/%s_%s_%s_false_neutral.csv' \
#                                  % (obj_weight_threshold, pos_score_threshold, neg_score_threshold), 'w'),\
#                            delimiter=',', quotechar='"', escapechar='\\')
    
    total_pos = 0
    total_neg = 0
    total_neu = 0
    predict_pos = 0
    predict_neg = 0
    predict_neu = 0
    err_pos = 0
    err_neg = 0
    err_neu = 0
    min_pos_score = 100
    max_pos_score = 0
    min_neg_score = 0
    max_neg_score = -100
    
    counter = 0
    for row in reader:
        # control how many tweets will be processed
        counter += 1
        if counter > 10000:
            break
        
        if row[1] == 'positive':
            total_pos += 1
        elif row[1] == 'negative':
            total_neg += 1
        else:
            total_neu += 1
        
        senti_score, obj_weight, words = analyze(row[4], features)
        
        if senti_score > pos_score_threshold and obj_weight < obj_weight_threshold:
            predict_pos += 1
            if senti_score > max_pos_score: max_pos_score = senti_score
            if senti_score < min_pos_score: min_pos_score = senti_score
            if row[1] != 'positive':
                err_pos += 1
                false_pos = [obj_weight, 'positive', senti_score, row[1], row[4], words]
#                pos_writer.writerow(false_pos)
#                print false_pos
        elif senti_score < neg_score_threshold and obj_weight < obj_weight_threshold:
            predict_neg += 1
            if senti_score > max_neg_score: max_neg_score = senti_score
            if senti_score < min_neg_score: min_neg_score = senti_score
            if row[1] != 'negative':
                err_neg += 1
                false_neg = [obj_weight,'negative', senti_score, row[1], row[4], words]
#                neg_writer.writerow(false_neg)
#                print false_neg
        else:
            predict_neu += 1
            if row[1] == 'positive' or row[1] == 'negative':
                err_neu += 1
                false_neu = [obj_weight,'neutral', senti_score, row[1], row[4], words]
#                neu_writer.writerow(false_neu)
#                print false_neu
    
#    print '===================== obj_weight [%s], pos_threshold [%s], neg_score_threshold [%s] =====================\n' \
#            % (obj_weight_threshold, pos_score_threshold, neg_score_threshold)
#    print '\nMax positive score: %.2f; min positive score: %.2f' % (max_pos_score, min_pos_score)
#    print '\nMax negative score: %.2f; min negative score: %.2f' % (max_neg_score, min_neg_score)
    true_pos = predict_pos - err_pos
    pos_p = float(true_pos) / predict_pos
    pos_r = float(true_pos) / total_pos
    pos_f = 2 * pos_p * pos_r / (pos_p + pos_r)
#    print '\nTotal positive: %d; false positive: %d ; precision: %.2f; recall: %.2f; f-measure: %.2f;' % \
#            (predict_pos, err_pos, pos_p, pos_r, pos_f)
    true_neg = predict_neg - err_neg
    neg_p = float(true_neg) / predict_neg
    neg_r = float(true_neg) / total_neg
    neg_f = 2 * neg_p * neg_r / (neg_p + neg_r)
#    print '\nTotal negative: %d; false negative: %d ; precision: %.2f; recall: %.2f; f-measure: %.2f;' % \
#            (predict_neg, err_neg, neg_p, neg_r, neg_f)
    true_neu = predict_neu - err_neu
    neu_p = float(true_neu) / predict_neu
    neu_r = float(true_neu) / total_neu
    neu_f = 2 * neu_p * neu_r / (neu_p + neu_r)
#    print '\nTotal neutral: %d; false neutral: %d ; precision: %.2f; recall: %.2f; f-measure: %.2f;' % \
#            (predict_neu, err_neu, neu_p, neu_r, neu_f)
#    print '\n'
    
    pos_data = [pos_p, pos_r, pos_f]
    neg_data = [neg_p, neg_r, neg_f]
    neu_data = [neu_p, neu_r, neu_f]
    return pos_data, neg_data, neu_data

def main():
    dispatch_pos()
    
    # retrieve adjective words
    
    # {word: [(sentiment, value)]} -- sentiment: 1 - pos, -1 - neg, 0 - obj
    all_words, sense_counter = map_words('senti_a.csv')
    map_writer = csv.writer(open('map_a.csv', 'w'), delimiter=',', quotechar='"', escapechar='\\')
    map_writer.writerow(['word', '(sentiment, value)'])
    for word in all_words:
        map_writer.writerow([word, all_words[word]])
    
    # {word: {'pos_socre': score, 'neg_score': score, 'post_weight': weight, 'neg_weight': weight, 'obj_weight': weight}}
    reduce_words(all_words, sense_counter)    
    reduce_writer = csv.writer(open('reduce_a.csv', 'w'), delimiter=',', quotechar='"', escapechar='\\')
    reduce_writer.writerow(['word', 'pos_score', 'neg_score', 'pos_weight', 'neg_weight', 'obj_weight'])
    for word in all_words:
        metrics = all_words[word]
        reduce_writer.writerow([word, metrics['pos_score'], metrics['neg_score'], \
                                metrics['pos_weight'], metrics['neg_weight'], metrics['obj_weight']])
    
#    # retrieve adverb words
#    
#    # {word: [(sentiment, value)]} -- sentiment: 1 - pos, -1 - neg, 0 - obj
#    all_words_r, sense_counter = map_words('senti_r.csv')
#    map_writer = csv.writer(open('map_r.csv', 'w'), delimiter=',', quotechar='"', escapechar='\\')
#    map_writer.writerow(['word', '(sentiment, value)'])
#    for word in all_words_r:
#        map_writer.writerow([word, all_words_r[word]])
#    
#    # {word: {'pos_socre': score, 'neg_score': score, 'post_weight': weight, 'neg_weight': weight, 'obj_weight': weight}}
#    reduce_words(all_words_r, sense_counter)    
#    reduce_writer = csv.writer(open('reduce_r.csv', 'w'), delimiter=',', quotechar='"', escapechar='\\')
#    reduce_writer.writerow(['word', 'pos_score', 'neg_score', 'pos_weight', 'neg_weight', 'obj_weight'])
#    for word in all_words_r:
#        metrics = all_words_r[word]
#        reduce_writer.writerow([word, metrics['pos_score'], metrics['neg_score'], \
#                                metrics['pos_weight'], metrics['neg_weight'], metrics['obj_weight']])
#    
#    all_words = dict(all_words_a.items() + all_words_r.items())
    
    # {word: (sentiment, score, weight, {<metrics>})}
    assign_sentiment(all_words)
    senti_writer = csv.writer(open('senti_words.csv', 'w'), delimiter=',', quotechar='"', escapechar='\\')
    senti_writer.writerow(['word', 'sentiment', 'score', 'weight', 'postings'])
    for word in all_words:
        senti_writer.writerow([word, all_words[word][0], all_words[word][1], all_words[word][2], all_words[word][3]])
    
    # {word: (sentiment, score, weight)}
    features = extract_features(all_words)
    feature_writer = csv.writer(open('features.csv', 'w'), delimiter=',', quotechar='"', escapechar='\\')
    feature_writer.writerow(['word', 'sentiment', 'score', 'weight'])
    features = sorted(features.iteritems(), key=lambda(word, senti): (senti, word)) # sorted function returns a list
    for word, senti in features:
        feature_writer.writerow([word, senti[0], senti[1], senti[2]])
    features = dict(features)
#    print features
    
#    classifier('sentiment.csv', features)
    
    results_writer = csv.writer(open('results.csv', 'w'), delimiter=',', quotechar='"', escapechar='\\')
    results_writer.writerow(['obj_weight_threshold', 'pos_score_threshold', 'neg_score_threshold', \
                                         'pos_data [p, r, f]', 'neg_data [p, r, f]', 'neu_data [p, r, f]', 'avg_data [p, r, f]'])
    results = {}
    max_pos_p = [[0],0]
    max_neg_p = [[0],0]
    max_neu_p = [[0],0]
    obj_weight_threshold = .01
    print '====== obj_weight_threshold: ', obj_weight_threshold
    while obj_weight_threshold < .2:
        results[obj_weight_threshold] = {}
        pos_score_threshold = .08
        print '==== pos_score_threshold: ', pos_score_threshold
        while pos_score_threshold < .2:
            results[obj_weight_threshold][pos_score_threshold] = {}
            neg_score_threshold = -.01
            print '== neg_score_threshold: ', neg_score_threshold
            while neg_score_threshold > -.2:
                pos_data, neg_data, neu_data = \
                        classifier('sentiment.csv', features, obj_weight_threshold, pos_score_threshold, neg_score_threshold)
                results[obj_weight_threshold][pos_score_threshold][neg_score_threshold] = (pos_data, neg_data, neu_data)
                avg_data = [sum(t) / 3 for t in zip(pos_data, neg_data, neu_data)]
                results_writer.writerow([obj_weight_threshold, pos_score_threshold, neg_score_threshold, \
                                         pos_data, neg_data, neu_data, avg_data])
                
                if pos_data[0] > max_pos_p[0][0]:
                    max_pos_p[0] = pos_data
                    max_pos_p[1] = [obj_weight_threshold, pos_score_threshold, neg_score_threshold]
                if neg_data[0] > max_neg_p[0][0]:
                    max_neg_p[0] = neg_data
                    max_neg_p[1] = [obj_weight_threshold, pos_score_threshold, neg_score_threshold]
                if neu_data[0] > max_neu_p[0][0]:
                    max_neu_p[0] = neu_data
                    max_neu_p[1] = [obj_weight_threshold, pos_score_threshold, neg_score_threshold]
                neg_score_threshold -= .01
                if neg_score_threshold * 100 % 5 == 0:
                    print '== neg_score_threshold: ', neg_score_threshold
            pos_score_threshold += .01
            if pos_score_threshold * 100 % 4 == 0:
                print '==== pos_score_threshold: ', pos_score_threshold
        obj_weight_threshold += .01
        if obj_weight_threshold * 100 % 5 == 0:
            print '====== obj_weight_threshold: ', obj_weight_threshold
    
    
    print '========================================================='
    print 'max_pos_p: %s -- [obj_weight_threshold: %.2f, pos_score_threshold: %.2f, neg_weigh_threshold: %.2f]' \
            % (map(lambda i: '%.2f' % i, max_pos_p[0]), max_pos_p[1][0], max_pos_p[1][1], max_pos_p[1][2])
    print 'max_neg_p: %s -- [obj_weight_threshold: %.2f, pos_score_threshold: %.2f, neg_weigh_threshold: %.2f]' \
            % (map(lambda i: '%.2f' % i, max_neg_p[0]), max_neg_p[1][0], max_neg_p[1][1], max_neg_p[1][2])
    print 'max_neu_p: %s -- [obj_weight_threshold: %.2f, pos_score_threshold: %.2f, neg_weigh_threshold: %.2f]' \
            % (map(lambda i: '%.2f' % i, max_neu_p[0]), max_neu_p[1][0], max_neu_p[1][1], max_neu_p[1][2])
    print '========================================================='

if __name__ == '__main__':
    main()

