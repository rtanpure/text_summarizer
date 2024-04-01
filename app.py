# -*- coding: utf-8 -*-
"""
Created on Sat march 20 15:09:52 2022

@author: rtanp
"""
import streamlit as st
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import networkx as nx


def welcome():
    return "Welcome all"

def text_summary(text_):
    
    sentences = []
    sentences.append(sent_tokenize(text_))
    sentences = [y for x in sentences for y in x]
    
    word_embeddings = {}
    f = open('glove.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ",regex=True)
    clean_sentences = [s.lower() for s in clean_sentences]
    stop_words = stopwords.words('english')
    def remove_stopwords(sen):
        sen_new = " ".join([i for i in sen if i not in stop_words])
        return sen_new
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    
    sentence_vectors = []
    for i in clean_sentences:
      if len(i) != 0:
        v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
      else:
        v = np.zeros((100,))
      sentence_vectors.append(v)
    
    sim_mat = np.zeros([len(sentences), len(sentences)])
    from sklearn.metrics.pairwise import cosine_similarity
    for i in range(len(sentences)):
      for j in range(len(sentences)):
        if i != j:
          sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
   
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)   
    
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    summarize_text=[]
    for i in range(5):
        summarize_text.append("".join(ranked_sentences[i][1]))
      
    return summarize_text
    
def main():
    st.title("Text Summarization")
    html_temp = """
    <div style="background-color:blue;padding:5px">
    <h2 style="color:white;text-align:center;">Text Summarization with NLTK \n (Data Science) App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    text_ = st.text_input("Input Text","")
    result=""
    if st.button("Summarise"):
        result=text_summary(text_)
    st.markdown("Summary:",)
    st.success('{}'.format(result))
    if st.button("About"):
        st.text("Made By Rishabh Tanpure")
        st.text("app built with Streamlit")
    
    
    
if __name__=='__main__':
    main()
