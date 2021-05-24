import torch
import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import collections
import string
import re
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
from thai2transformers import preprocess
from thai2transformers.preprocess import process_transformers

#Parameters:
param_smooth = 10
param_threshold = 0.9
param_span = 4

param_backoff_limit = 3

def read_text_file(filename):
    f = open("/content/drive/Shareddrives/Pretend To Understand NLP/"+filename, "r", encoding="utf-8")
    d_ = []
    for x in f:
        d_.append(x.strip())
    return d_

def read_all_datasets():
    d_pos = read_text_file('pos_post_process_all.txt')
    d_neg = read_text_file('neg_post_process_all.txt')

    tcas_pos = read_text_file('tcas61_pos.txt')
    tcas_neg = read_text_file('tcas61_neg.txt')

    review_pos = read_text_file('review_shopping_pos.txt')
    review_neg = read_text_file('review_shopping_neg.txt')

    general_pos = read_text_file('general_pos.txt')
    general_neg = read_text_file('general_neg.txt')

    all_pos = d_pos + tcas_pos + review_pos + general_pos
    all_neg = d_neg + tcas_neg + review_neg + general_neg

    return all_pos, all_neg


tokenizer = AutoTokenizer.from_pretrained('airesearch/wangchanberta-base-att-spm-uncased',
                                                revision='finetuned@wisesight_sentiment')

model = AutoModelForSequenceClassification.from_pretrained('airesearch/wangchanberta-base-att-spm-uncased',
                                                        revision='finetuned@wisesight_sentiment',
                                                        output_attentions=False)
classify_multiclass = pipeline(task='sentiment-analysis',
         tokenizer=tokenizer,
         model = model)


##################################
##### Frequency Ratio Method #####

##### N-granms ###################
#ngram has punctuation
def has_punctuation(ngram): 
    return True in [x in string.punctuation for x in ngram]

def generate_ngrams(lines, min_length=1, max_length=param_span):
    lengths = range(min_length, max_length + 1)
    ngrams = {length: [] for length in lengths}
    queue = collections.deque(maxlen=max_length)
    
    def add_queue():
        current = tuple(queue)
        for length in lengths:
            if len(current) >= length and not has_punctuation(current[:length]) and current[:length] not in ngrams[length]:
                ngrams[length].append(current[:length])
    
    short_by = 0
    for line in lines:
        short_by = max(0, max_length - len(lines))
        for word in tokenizer.tokenize(process_transformers(line)):
            queue.append(word)
            if len(queue) >= max_length-short_by:
                add_queue()                

    while len(queue) > min_length:
        queue.popleft()
        add_queue()
    return ngrams

#modified from & fixed their error of ngram with # of words < 4: https://gist.github.com/benhoyt/dfafeab26d7c02a52ed17b6229f0cb52
def count_ngrams(lines, min_length=1, max_length=param_span):
    """Iterate through given lines iterator (file object or list of
    lines) and return n-gram frequencies. The return value is a dict
    mapping the length of the n-gram to a collections.Counter
    object of n-gram tuple and number of times that n-gram occurred.
    Returned dict includes n-grams of length min_length to max_length.
    """
    lengths = range(min_length, max_length + 1)
    ngrams = {length: collections.Counter() for length in lengths}
    queue = collections.deque(maxlen=max_length)

    # Helper function to add n-grams at start of current queue to dict
    def add_queue():
        current = tuple(queue)
        for length in lengths:
            if len(current) >= length and not has_punctuation(current[:length]):
                ngrams[length][current[:length]] += 1

    # Loop through all lines and words and add n-grams to dict
    short_by = 0
    for line in lines:
        short_by = max(0, max_length - len(lines))
        # for word in line.split():
        for word in tokenizer.tokenize(process_transformers(line)):
            queue.append(word)
            if len(queue) >= max_length - short_by:
                add_queue()

    # Make sure we get the n-grams at the tail end of the queue
    while len(queue) > min_length:
        queue.popleft()
        add_queue()

    return ngrams

def get_counts(list1, counted_ngrams):
    counts = []
    list1_ngrams = generate_ngrams(list1)
    # print(list1_ngrams)
    list2_counts = counted_ngrams
    
    for length in range(param_span,0, -1):
        for v in list1_ngrams[length]:
            counts.append([list2_counts[length][v], v])
    return np.array(counts)

class ngrams_counts():
    global read_all_datasets, count_ngrams
    def __init__(self):
        all_pos, all_neg = read_all_datasets() 
        self.pos_ngrams = count_ngrams(all_pos)
        self.neg_ngrams = count_ngrams(all_neg)

##### Getting Attribute and Context #########

#these are methods that will become useful when extracting attribute markers
#why do we need all this? well... that's like 5 hours of debugging...
def flatten(foo):
    return list(_flatten(foo))

def _flatten(foo):
    for x in foo:
        if isinstance(x, collections.Iterable) and not isinstance(x, str):
            for y in _flatten(x):
                yield y
        else:
            yield x

def array_to_string(a):
    return ' '.join(flatten(a))

def is_in_string_array(elements, original): #deprecated, does not take into account sequence order
    the_elements = [x for x in array_to_string(elements).split() if x != '▁']
    the_original = array_to_string(original).split()
    # print('test:',the_elements,the_original,np.isin(the_elements, the_original).any())
    return np.isin(the_elements, the_original).any()

def insert_string(string, inserted_string, index):
    return string[:index] + inserted_string + string[index:]

# modified from https://stackoverflow.com/questions/41752946/replacing-a-character-from-a-certain-index
def replace_string(s, newstring, index, nofail=False):
    # raise an error if index is outside of the string
    if not nofail and index not in range(len(s)):
        raise ValueError("index outside given string. index:" + index)

    # if not erroring, but the index is still not in the correct range..
    if index < 0:  # add it to the beginning
        return newstring + s
    if index > len(s):  # add it to the end
        return s + newstring

    # insert the new string between "slices" of the original
    return s[:index] + newstring + s[index + len(newstring):]

def get_attribute_markers(s, style_src, pos_ngrams_counts, neg_ngrams_counts):
    sentence = [s]
        
    pos_counts = get_counts(sentence, pos_ngrams_counts)
    pythai_pos_counts = get_counts
    pos_ngrams = pos_counts[:,1]
    if len(pos_counts) > 0:
        pos_counts = pos_counts[:,0]
    
    neg_counts = get_counts(sentence, neg_ngrams_counts)
    neg_ngrams = neg_counts[:,1]
    if len(neg_counts) > 0:
        neg_counts = neg_counts[:,0]
    
    label = 'neg'
    if(style_src):
        importances = (pos_counts + param_smooth) / (neg_counts + param_smooth)
        ngrams = pos_ngrams
        label = 'pos'
    else:
        importances = (neg_counts + param_smooth) / (pos_counts + param_smooth)
        ngrams = neg_ngrams
        
    a = []

    importances = np.vstack((importances, ngrams)).T

    for importance in sorted(importances,key=lambda x:(x[0],-len(x[1])),reverse=True):
        if importance[0] > param_threshold and not is_in_string_array(importance[1], a) and classify_multiclass(' '.join(importance[1]))[0]['label']==label:
            a.append(' '.join(importance[1]))

    return a

def separate(sentence, style_src, pos_ngrams_counts, neg_ngrams_counts):
    attributes = get_attribute_markers(sentence, style_src, pos_ngrams_counts, neg_ngrams_counts)
    c = sentence

    replace_indexes = []
    for a in attributes:
        a_striped = a.strip('▁').replace(' ','')
        replace_index = -1
        replace_index = c.find(a_striped)
        replace_indexes.append(replace_index)
        if a_striped != "":
          c = c.replace(a_striped, "<mask>")
  
    if len(attributes) == 0:
        return {'c': c, 'a': [], 'i': [], 's': sentence}
    
    replace_indexes, attributes = zip(*sorted(zip(replace_indexes, attributes)))
    return {'c': c, 'a': attributes, 'i': replace_indexes, 's': sentence}

def get_c_and_a(sentence, style_src, pos_ngrams_counts, neg_ngrams_counts):
    sep_res = separate(sentence, style_src, pos_ngrams_counts, neg_ngrams_counts)
    c = re.sub(' +', ' ', sep_res['c'])
    a = sep_res['a']
    return c,a

def pos_or_neg(src_sentence):
    # src_class = classify_multiclass(src_sentence)
    # print("msg = ", src_sentence)
    # print("src = ", src_class)
    # return src_class[0]['label']
    return 'neg'

def delete_negative(sentence, pos_ngrams_counts, neg_ngrams_counts):
    print("class = ", classify_multiclass(sentence)[0].get('label'))
    if classify_multiclass(sentence)[0].get('label') != 'neg':
        return 'Pos'
    else:
        sep_res = separate(sentence, 0, pos_ngrams_counts, neg_ngrams_counts)
        c = re.sub(' +', ' ', sep_res['c'])
        a = sep_res['a']
        return a,c,sentence

# import delete_module as delete
# ngrams = delete.ngrams_counts()
# delete_output = delete.delete_negative(src_sentence, ngrams.pos_ngrams, ngrams.neg_ngrams)
### output example: (('แย่มาก',), 'ร้านนี้<mask>', 'ร้านนี้แย่มาก')
