'''
Created on Mar 12, 2012

@author: William
'''

import urllib
import json
import sys
import codecs
import time

def load_tweets(q):

    url = 'http://search.twitter.com/search.json?q=%s&lang=en&include_entities=true&result_type=mixed&since_id='
    
    count = 1
    rtf = codecs.open('raw_tweets_%d.txt' % count, 'a', encoding='utf-8')
    tf = codecs.open('simple_tweets.txt', 'a', encoding='utf-8')
    url = url % q
    
    max_id = '' # update this after loading tweets encounters error
    if max_id:
        next_url = url + max_id
    
    while True:
        
        if count % 500 == 0:
            rtf.close()
            rtf = codecs.open('raw_tweets_%d.txt' % count, 'a', encoding='utf-8')
        count += 1
        
        print '>>>>>> URL:', next_url
        
        try:
            ret = urllib.urlopen(next_url).read()
            rtf.write(ret + '\n')
            rtf.flush()
            
            ret = json.loads(ret)
        except ValueError:
            if ret.find('503 Service Unavailable') != -1:
                print 'Got 503 -- wait for 5 seconds and retry...'
                time.sleep(5)
                continue
            print '*** Error info ================================'
            print sys.exc_info()[0]
            break
        except:
            print '*** Return ===================================='
            print ret
            print '*** Error info ================================'
            print sys.exc_info()[0]
            break
        
        max_id = ret['max_id']
        results = ret['results']
        
        for tweet_entity in results:
            tweet = tweet_entity['text']
            print tweet
            tf.write(tweet + '\n')
            tf.flush()
        
        time.sleep(5)
    
    tf.close()
    rtf.close()
    return ret
        