# !pip install pythainlp

from pythainlp import word_tokenize # tokenizer
from pythainlp.word_vector import * # thai2vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

def sentence_vectorizer(ss,dim=300,use_mean=True):
    s = word_tokenize(ss)
    vec = np.zeros((1,dim))
    for word in s:
        if word in model.index2word:
            vec+= model.word_vec(word)
        else: pass
    if use_mean: vec /= len(s)
    return vec

class initialize():
    def __init__(self):
        global read_all_datasets, sentence_vectorizer, model
        model = get_model()
        pos_attr, pos_con_masked, pos_ref, pos_con,neg_attr, neg_con_masked, neg_ref, neg_con = read_all_datasets()
        pos_vector = []
        for e in pos_ref:
            pos_vector.append(sentence_vectorizer(e)[0])
        self.pos_vector = np.array(pos_vector)
        self.pos_attr = pos_attr

def retrieve_output(input, pos_vector, pos_attr):
    input_content = delete_mask([input[1]])[0]
    cosine_sim = cosine_similarity(sentence_vectorizer(input_content), pos_vector)[0]
    closest_attr_idx = np.argmax(cosine_sim)
    return [input[2], input[1], pos_attr[closest_attr_idx]]

# import retrieve_module_thai2vec as retrieve
# init = retrieve.initialize()
# retrieve_output = retrieve.retrieve_output(delete_output, init.pos_vector, init.pos_attr)
### output example: ['ร้านนี้แย่มาก', 'ร้านนี้<mask>', ['ป้ะ']]
