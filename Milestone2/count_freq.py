from string import punctuation
from collections import defaultdict
import csv
import nltk
from nltk.corpus import stopwords

mywords = {}

def count_freq(file_name, N):
    words = {}
    # stopwords = nltk.corpus.stopwords.words('english')
    reader = csv.reader( open(file_name, 'rb'), delimiter=',', quotechar='"', escapechar='\\' )
    
    words_gen = []
    for line in reader:
        words_in_line = []
        words_in_line += [word.strip(punctuation).lower() for word in line[4].split(' ') if word]
        for i in range(len(words_in_line)):
            words_gen.append(words_in_line[i]) # + ' ' + words_in_line[i + 1] + ' ' + words_in_line[i + 2])
    # print words_gen
    
    words = defaultdict(int)
    for word in words_gen:
        words[word] +=1
    
    top_words = sorted(words.iteritems(),
                       key=lambda(word, count): (-count, word))[:N] 
    
    return top_words
    
#    for word, frequency in top_words:
#        print "%s: %d" % (word, frequency)


#print count_freq('sentiment1.csv', 100)