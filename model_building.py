from flask import Flask, render_template, request, url_for, flash, redirect
from selenium.webdriver import Chrome, Firefox
from selenium.common.exceptions import NoSuchElementException

from selenium import webdriver
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.common.keys import Keys
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model2 = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4, epochs=20)
def calculate_similarity(doc1, doc2):
    vec1 = model2.infer_vector(simple_preprocess(doc1))
    vec2 = model2.infer_vector(simple_preprocess(doc2))
    similarity_score = cosine_similarity([vec1], [vec2])[0][0]
    return similarity_score

#open a python file and write the code for building and training model of doctovec and get model
# get the data from the databse downladed and split into train and test
documents_list = []
documents = documents_list
#train data to train here
'''for name, docs_dicts in data_all_searches.items():
    for doc_dict in docs_dicts:
        for i in range(0,50):
            documents.append(doc_dict[i]['paragrahs'])'''


# Preprocess the documents (simple_preprocess tokenizes the text)
preprocessed_documents = [simple_preprocess(doc) for doc in documents]

# Create TaggedDocument for Doc2Vec training
tagged_data = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(preprocessed_documents)]

# Initialize and train the Doc2Vec model
model2.build_vocab(tagged_data)
model2.train(tagged_data, total_examples=model2.corpus_count, epochs=model2.epochs)


## end train

## test data
#infer the model and get embeddings for one kind of class data  print the similarity scores and then print the data

# Sample documents for similarity calculation

## in test data also get seed documents data and then write a for loop for df's document and say predict similarity scores
document1 = "This is a document to compare similarity."
document2 = "This document is different from the first one."

# Calculate similarity scores
similarity_score1 = calculate_similarity(document1, document2)
print(f"Similarity Score between Document 1 and Document 2: {similarity_score1:.4f}")
## end testing
#compare those scores to that of the bert

#output comparission
