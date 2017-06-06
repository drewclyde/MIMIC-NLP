import numpy as np
import pandas as pd
import MySQLdb
import re
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize


db = MySQLdb.connect(host="0.0.0.0",    # your host, usually localhost
                     user="user",         # your username
                     passwd="passwd",  # your password
                     db="mimic_sql")        # name of the data base

physician_df = pd.read_sql("SELECT ROW_ID, CATEGORY, TEXT FROM NOTEEVENTS where category='Physician' limit 2670", con=db)
sw_df = pd.read_sql("SELECT ROW_ID, CATEGORY, TEXT FROM NOTEEVENTS where category='Social Work'", con=db)
notes = pd.concat([physician_df, sw_df])

label_sentences = []
max_length = 0

for index, row in notes.iterrows():
    line_array = sent_tokenize(row['TEXT'])
    if len(line_array) > max_length:
        max_length = len(line_array)
    rn = row['ROW_ID']
    for sentence in line_array:
        an = line_array.index(sentence)
        sentence = re.sub('[^A-Za-z\s]+', ' ', sentence).lower().replace("\n"," ").split()
        if sentence != []:
            label_sentences.append(LabeledSentence(sentence, [str(rn) + '-' + str(an)]))


model = Doc2Vec(min_count=5, window=10, size=200, sample=1e-4, negative=5, workers=8)

model.build_vocab(label_sentences)


fname = 'd2v-200'
model.save(fname)

model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

del db
del notes
del label_sentences


