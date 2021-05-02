# !pip install pythainlp
# !pip install nltk
# !pip install scikit-learn

import pandas as pd
import numpy as np
import pythainlp
from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.corpus import wordnet
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity

def clean_attr(attr_list):
  out = []
  for sublist in attr_list:
    tmp = []
    for e in sublist:
      e = e.replace('<_>', '')
      e = e.replace('▁', '')
      e = e.replace(' ', '')
      tmp.append(e)
    out.append(tmp)
  return out

def read_text_file(filename):
  f = open(f'./data/{filename}', 'r')
  attr = []
  con = []
  ref = []
  for e in f:
    # 1input: attribute, content, original reference sentence
    attr_list = []
    e = e[2:-2]
    if e[0] != ']':
      attribute, x = e.split('),')
      tmp = attribute.split(',')
      for e in tmp:
        e = e.strip()
        e = e.strip("'")
        e = e.strip()
        if e != '':
          attr_list.append(e)
      attr.append(attr_list)
      content, ref_sentence = x.strip().split(', ')
      con.append(content.strip("'"))
      ref.append(ref_sentence.strip("'"))
  attr = clean_attr(attr)
  return attr, con, ref

def delete_mask(content_list):
  con_out = []
  for i in range (len(content_list)):
    con = content_list[i].replace('<mask>', '')
    con_out.append(con.strip())
  return con_out

def read_all_datasets():
    pythai_neg_attr, pythai_neg_con_masked, pythai_neg_ref = read_text_file('delete_output_pythai_v2.txt')
    wisesight_neg_attr, wisesight_neg_con_masked, wisesight_neg_ref = read_text_file('delete_output_v2.txt')
    pythai_pos_attr, pythai_pos_con_masked, pythai_pos_ref = read_text_file('delete_output_pythai_pos_v2.txt')
    wisesight_pos_attr, wisesight_pos_con_masked, wisesight_pos_ref = read_text_file('delete_output_wisight_pos_v2.txt')

    pos_attr = pythai_pos_attr + wisesight_pos_attr
    pos_con_masked = pythai_pos_con_masked + wisesight_pos_con_masked
    pos_ref = pythai_pos_ref + wisesight_pos_ref
    neg_attr = pythai_neg_attr + wisesight_neg_attr
    neg_con_masked = pythai_neg_con_masked + wisesight_neg_con_masked
    neg_ref = pythai_neg_ref + wisesight_neg_ref
    pos_con = delete_mask(pos_con_masked)
    neg_con = delete_mask(neg_con_masked)
    return pos_attr, pos_con_masked, pos_ref, pos_con,\
    neg_attr, neg_con_masked, neg_ref, neg_con

def tokenize(text):
    tokens = word_tokenize(text,engine='newmm')
    # Remove stop words
    tokens = [i for i in tokens if not i in thai_stopwords()]
    # Remove numeric text
    tokens = [i for i in tokens if not i.isnumeric()]
    # Remove space
    tokens = [i for i in tokens if not ' ' in i]
    # Stemmer: EN
    tokens = [p_stemmer.stem(i) for i in tokens]
    # Stemmer: TH
    tokens_temp=[]
    for i in tokens:
        w_syn = wordnet.synsets(i)
        if (len(w_syn)>0) and (len(w_syn[0].lemma_names('tha'))>0):
            tokens_temp.append(w_syn[0].lemma_names('tha')[0])
        else:
            tokens_temp.append(i)
    tokens = tokens_temp
    return tokens

def tf_idf(tok_neg_con, tok_pos_con):
  tokenized = tok_neg_con + tok_pos_con
  tok_joined = [','.join(tkn) for tkn in tokenized]
  tfidf = TfidfVectorizer(analyzer=lambda x:x.split(','),)
  vec = tfidf.fit_transform(tok_joined)
  vec_neg = vec[:len(neg_con)]
  vec_pos = vec[len(neg_con):]
  return tfidf,vec_neg,vec_pos

class initialize():
    global read_all_datasets, tokenize, tf_idf, p_stemmer
    def __init__(self):
        nltk.download('words')
        p_stemmer = PorterStemmer()
        pos_attr, pos_con_masked, pos_ref, pos_con,neg_attr, neg_con_masked, neg_ref, neg_con = read_all_datasets()
        tok_neg_con = [tokenize(e) for e in neg_con]
        tok_pos_con = [tokenize(e) for e in pos_con]
        tfidf,tfidf_neg, tfidf_pos = tf_idf(tok_neg_con, tok_pos_con)
        self.tfidf = tfidf
        self.pos_attr = pos_attr
        self.tfidf_pos = tfidf_pos

def tf_idf_con(input_content,tfidf):
  tokens = tokenize(input_content)
  tok_joined = [','.join(tkn) for tkn in tokens]
  vec = tfidf.transform(tok_joined)
  return vec

def retrieve_output(input, tfidf_feat, tfidf_pos, pos_attr):
    ### del output example: (('แย่มาก',), 'ร้านนี้<mask>', 'ร้านนี้แย่มาก')
    input_content = delete_mask([input[1]])[0]
    tfidf = tf_idf_con(input_content,tfidf_feat)
    cosine_sim = cosine_similarity(tfidf, tfidf_pos)[0]
    closest_attr_idx = np.argmax(cosine_sim)
    return [input[2], input[1], pos_attr[closest_attr_idx]]

# import retrieve_module_tfidf as retrieve
# init = retrieve.initialize()
# retrieve_output = retrieve.retrieve_output(delete_output, init.tfidf, init.tfidf_pos, init.pos_attr)
### output example: ['ร้านนี้แย่มาก', 'ร้านนี้<mask>', ['น่าสน']]